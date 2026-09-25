from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from .egypt_id import EgyptIdValidation, extract_valid_national_id, validate_egyptian_national_id
from .extractor import HybridExtractor, PaddleOcrExtractor, PaddleOcrVlExtractor, QwenOpenVinoExtractor


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_OCR_DIMENSION = 2000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
UPLOAD_CHUNK_SIZE = 1024 * 1024

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
logger = logging.getLogger(__name__)

app = FastAPI(title="Egyptian National ID OCR", version="0.1.0")
OCR_ENGINE = os.getenv("AIDOC_OCR_ENGINE", "classic").lower()
if OCR_ENGINE == "vl":
    extractor = PaddleOcrVlExtractor(pipeline_version="v1.6")
elif OCR_ENGINE in {"qwen", "qwen-vl", "openvino", "hybrid", "classic-qwen", "qwen-fallback"}:
    extractor = QwenOpenVinoExtractor(
        model_path=os.getenv(
            "QWEN_VL_MODEL_PATH",
            "models/Qwen3-VL-4B-Instruct-int4-ov",
        ),
        device=os.getenv("QWEN_VL_DEVICE", "CPU"),
        max_new_tokens=int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "300")),
    )
elif OCR_ENGINE in {"hybrid", "classic-qwen", "qwen-fallback"}:
    extractor = HybridExtractor(
        classic=PaddleOcrExtractor(lang=os.getenv("AIDOC_OCR_LANG", "ar")),
        qwen=QwenOpenVinoExtractor(
            model_path=os.getenv(
                "QWEN_VL_MODEL_PATH",
                "models/Qwen3-VL-4B-Instruct-int4-ov",
            ),
            device=os.getenv("QWEN_VL_DEVICE", "CPU"),
            max_new_tokens=int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "300")),
        ),
        always_run_qwen=os.getenv("HYBRID_ALWAYS_QWEN", "0") == "1",
    )
else:
    extractor = PaddleOcrExtractor(lang=os.getenv("AIDOC_OCR_LANG", "ar"))


class VerificationResponse(BaseModel):
    valid: bool
    name: str | None
    national_id: str | None
    validation: EgyptIdValidation
    ocr_text: str
    confidence: float | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "ocr_engine": OCR_ENGINE, "qwen_model_path": os.getenv("QWEN_VL_MODEL_PATH", "") if OCR_ENGINE in {"qwen", "qwen-vl", "openvino", "hybrid", "classic-qwen", "qwen-fallback"} else ""}


async def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        total = 0
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="Image is too large. Upload an image smaller than 8 MB.",
                )
            tmp.write(chunk)
    return tmp_path


def prepare_image_for_ocr(source_path: Path) -> Path:
    try:
        with Image.open(source_path) as image:
            image.verify()
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((MAX_OCR_DIMENSION, MAX_OCR_DIMENSION))
            normalized = image.convert("RGB")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                ocr_path = Path(tmp.name)
                normalized.save(tmp, format="JPEG", quality=90, optimize=True)
                return ocr_path
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc


@app.post("/verify", response_model=VerificationResponse)
async def verify(file: UploadFile = File(...)) -> VerificationResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")

    upload_path = await save_upload(file)
    ocr_path: Path | None = None

    try:
        ocr_path = prepare_image_for_ocr(upload_path)
        fields = extractor.extract(ocr_path)
        national_id = extract_valid_national_id(
            "\n".join(part for part in (fields.national_id, fields.ocr_text) if part)
        )
        validation = validate_egyptian_national_id(national_id)
        return VerificationResponse(
            valid=validation.valid and bool(fields.name),
            name=fields.name,
            national_id=national_id,
            validation=validation,
            ocr_text=fields.ocr_text,
            confidence=fields.confidence,
        )
    except RuntimeError as exc:
        logger.exception("OCR runtime error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected OCR error")
        raise HTTPException(status_code=500, detail=f"Unexpected OCR error: {exc}") from exc
    finally:
        upload_path.unlink(missing_ok=True)
        if ocr_path is not None:
            ocr_path.unlink(missing_ok=True)
