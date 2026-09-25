from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .egypt_id import extract_national_id, extract_valid_national_id


ARABIC_LINE_RE = re.compile(r"[\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})+")
LABEL_WORDS = {
    "بطاقة",
    "تحقيق",
    "شخصية",
    "جمهورية",
    "مصر",
    "العربية",
    "الرقم",
    "القومي",
    "العنوان",
    "تاريخ",
    "الميلاد",
}
LABEL_WORDS_NORMALIZED = {
    "بطاقة",
    "تحقيق",
    "شخصية",
    "جمهورية",
    "مصر",
    "العربية",
    "الرقم",
    "القومي",
    "العنوان",
    "تاريخ",
    "الميلاد",
}
ADDRESS_START_WORDS = {
    "ش",
    "شارع",
    "حارة",
    "منزل",
    "رقم",
    "دور",
    "اول",
    "ثان",
    "قسم",
    "حي",
    "عين",
}
HEADER_PREFIXES = ("جمهور", "حمه", "جمهر", "مصر", "مص", "عرب", "حرب")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")




@dataclass(frozen=True)
class OcrRegion:
    name: str
    box: tuple[float, float, float, float]
    scale: int = 2
    contrast: float = 2.0

@dataclass(frozen=True)
class ExtractedFields:
    name: str | None
    national_id: str | None
    ocr_text: str
    confidence: float | None = None


def normalize_arabic_word(word: str) -> str:
    word = ARABIC_DIACRITICS_RE.sub("", word)
    word = word.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    word = word.replace("ى", "ي").replace("ة", "ه")
    return word


def normalize_arabic_name(name: str) -> str:
    words = name.split()
    if words and normalize_arabic_word(words[0]) == "احمد":
        words[0] = "أحمد"
    return " ".join(words)


def looks_like_header(words: list[str]) -> bool:
    normalized = [normalize_arabic_word(word) for word in words]
    matches = sum(
        1
        for word in normalized[:5]
        if any(word.startswith(prefix) for prefix in HEADER_PREFIXES)
    )
    return matches >= 2


def trim_name_words(words: list[str]) -> list[str]:
    trimmed: list[str] = []
    for word in words:
        normalized = normalize_arabic_word(word)
        if normalized in ADDRESS_START_WORDS:
            break
        if len(normalized) < 2:
            continue
        trimmed.append(word)
        if len(trimmed) == 5:
            break
    return trimmed


def name_words_after_label(words: list[str]) -> list[str]:
    normalized = [normalize_arabic_word(word) for word in words]
    label_indexes = [
        index
        for index, word in enumerate(normalized)
        if word in LABEL_WORDS_NORMALIZED
    ]
    if not label_indexes:
        return []
    return trim_name_words(words[label_indexes[-1] + 1 :])


def extract_name_from_text(text: str) -> str | None:
    lines = [line.strip(" :|-") for line in text.splitlines() if line.strip()]
    single_words: list[str] = []
    candidates: list[str] = []

    for line in lines:
        arabic_words = re.findall(r"[\u0600-\u06ff]{1,}", line)
        if not arabic_words:
            continue

        words_after_label = name_words_after_label(arabic_words)
        if len(words_after_label) >= 3:
            return normalize_arabic_name(" ".join(words_after_label))

        normalized_words = {normalize_arabic_word(word) for word in arabic_words}
        if normalized_words & LABEL_WORDS_NORMALIZED:
            continue
        if looks_like_header(arabic_words):
            continue

        if len(arabic_words) == 1:
            single_words.append(arabic_words[0])
            continue

        trimmed_words = trim_name_words(arabic_words)
        if len(trimmed_words) >= 3:
            for first_name in reversed(single_words):
                if normalize_arabic_word(trimmed_words[-1]) == normalize_arabic_word(first_name):
                    return normalize_arabic_name(" ".join([first_name, *trimmed_words[:-1]][:3]))
            candidates.append(" ".join(trimmed_words))

    return normalize_arabic_name(candidates[0]) if candidates else None


ID_REGIONS = (
    OcrRegion("national_id", (0.42, 0.62, 0.98, 0.82), 2, 2.0),
    OcrRegion("national_id", (0.00, 0.60, 1.00, 0.84), 2, 2.0),
)
NAME_REGIONS = (
    OcrRegion("name", (0.38, 0.24, 0.98, 0.50), 2, 1.8),
)
ADDRESS_REGIONS = (
    OcrRegion("address", (0.36, 0.40, 0.98, 0.64), 2, 1.8),
)


def make_region_images(image_path: Path, regions: tuple[OcrRegion, ...]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        for region_index, region in enumerate(regions):
            left, top, right, bottom = region.box
            crop = image.crop(
                (
                    int(width * left),
                    int(height * top),
                    int(width * right),
                    int(height * bottom),
                )
            )
            crop = crop.resize((crop.width * region.scale, crop.height * region.scale))
            grayscale = ImageOps.grayscale(crop)
            variants = (
                ImageEnhance.Contrast(grayscale).enhance(region.contrast).filter(ImageFilter.SHARPEN),
            )
            for variant_index, variant in enumerate(variants):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"-{region.name}-{region_index}-{variant_index}.png",
                ) as tmp:
                    variant.save(tmp, format="PNG")
                    paths.append((region.name, Path(tmp.name)))
    return paths


class PaddleOcrExtractor:
    def __init__(self, lang: str = "ar", text_det_limit_side_len: int = 960) -> None:
        self.lang = lang
        self.text_det_limit_side_len = text_det_limit_side_len
        self._pipeline = None

    def extract(self, image_path: Path) -> ExtractedFields:
        full_text = self._run_paddleocr(image_path)
        region_texts: dict[str, list[str]] = {"name": [], "address": [], "national_id": []}
        region_paths: list[tuple[str, Path]] = []
        try:
            region_paths = make_region_images(
                image_path,
                NAME_REGIONS + ADDRESS_REGIONS + ID_REGIONS,
            )
            for region_name, region_path in region_paths:
                region_text = self._run_paddleocr(region_path)
                if region_text:
                    region_texts.setdefault(region_name, []).append(region_text)
        finally:
            for _, region_path in region_paths:
                region_path.unlink(missing_ok=True)

        name_text = "\n".join(region_texts["name"])
        id_text = "\n".join(region_texts["national_id"])
        combined_text = "\n".join(
            part
            for part in (
                "[name regions]",
                name_text,
                "[address regions]",
                "\n".join(region_texts["address"]),
                "[national id regions]",
                id_text,
                "[full card]",
                full_text,
            )
            if part
        )

        return ExtractedFields(
            name=extract_name_from_text(name_text) or extract_name_from_text(full_text),
            national_id=extract_national_id(id_text) or extract_national_id(combined_text),
            ocr_text=combined_text,
        )

    def _run_paddleocr(self, image_path: Path) -> str:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddlepaddle and paddleocr in the runtime environment."
            ) from exc

        if self._pipeline is None:
            self._pipeline = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=self.text_det_limit_side_len,
            )

        results = self._pipeline.predict(str(image_path))
        return flatten_paddleocr_results(results)


QWEN_ID_PROMPT = """
You are reading the front side of an Egyptian national ID card.
Return JSON only, without Markdown:
{
  "name": string or null,
  "national_id_candidates": [strings],
  "address": string or null
}
Rules:
- Egyptian national_id must be exactly 14 digits.
- The national ID is usually printed as Arabic or English digits near the lower front side of the card.
- Include every possible 14-digit candidate you can see in national_id_candidates.
- Do not return passport, serial, watermark, phone model, or short code values like KW4749408 as national_id candidates.
- If the 14-digit Egyptian national ID is not visible, return an empty candidates array.
- Preserve Arabic names and addresses as written.
- Do not explain.
""".strip()


def json_block_from_text(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class QwenOpenVinoExtractor:
    def __init__(self, model_path: str, device: str = "CPU", max_new_tokens: int = 300) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def extract(self, image_path: Path) -> ExtractedFields:
        text = self._run_qwen(image_path)
        data = json_block_from_text(text)
        candidates = data.get("national_id_candidates")
        if isinstance(candidates, list):
            candidate_text = "\n".join(str(candidate) for candidate in candidates)
        else:
            candidate_text = ""

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            name = extract_name_from_text(text)

        address = data.get("address")
        ocr_text = "\n".join(
            part
            for part in (
                "[qwen json]",
                json.dumps(data, ensure_ascii=False) if data else "",
                "[qwen raw]",
                text,
                "[qwen address]",
                address if isinstance(address, str) else "",
            )
            if part
        )
        return ExtractedFields(
            name=normalize_arabic_name(name.strip()) if isinstance(name, str) else None,
            national_id=extract_national_id(candidate_text or text),
            ocr_text=ocr_text,
        )

    def _run_qwen(self, image_path: Path) -> str:
        try:
            import openvino as ov
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError(
                "Qwen OpenVINO runtime is not installed. Install openvino, openvino-genai, "
                "openvino-tokenizers, pillow, and numpy in the runtime environment."
            ) from exc

        if self._pipeline is None:
            if not Path(self.model_path).exists():
                raise RuntimeError(f"Qwen OpenVINO model path does not exist: {self.model_path}")
            self._pipeline = ov_genai.VLMPipeline(self.model_path, self.device)

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_data = np.array(image).reshape(1, image.height, image.width, 3).astype(np.uint8)

        result = self._pipeline.generate(
            QWEN_ID_PROMPT,
            images=[ov.Tensor(image_data)],
            max_new_tokens=self.max_new_tokens,
        )
        return getattr(result, "text", str(result))


class HybridExtractor:
    def __init__(
        self,
        classic: PaddleOcrExtractor,
        qwen: QwenOpenVinoExtractor,
        always_run_qwen: bool = False,
    ) -> None:
        self.classic = classic
        self.qwen = qwen
        self.always_run_qwen = always_run_qwen

    def extract(self, image_path: Path) -> ExtractedFields:
        classic_fields = self.classic.extract(image_path)
        classic_valid_id = extract_valid_national_id(
            "\n".join(
                part for part in (classic_fields.national_id, classic_fields.ocr_text) if part
            )
        )
        should_run_qwen = self.always_run_qwen or classic_valid_id is None or not classic_fields.name

        if not should_run_qwen:
            return ExtractedFields(
                name=classic_fields.name,
                national_id=classic_valid_id,
                ocr_text="\n".join(("[classic]", classic_fields.ocr_text)),
                confidence=classic_fields.confidence,
            )

        qwen_fields = self.qwen.extract(image_path)
        combined_text = "\n".join(
            part
            for part in (
                "[classic]",
                classic_fields.ocr_text,
                "[qwen]",
                qwen_fields.ocr_text,
            )
            if part
        )
        best_id = extract_valid_national_id(
            "\n".join(
                part
                for part in (
                    qwen_fields.national_id,
                    classic_valid_id,
                    combined_text,
                )
                if part
            )
        )
        return ExtractedFields(
            name=qwen_fields.name or classic_fields.name,
            national_id=best_id,
            ocr_text=combined_text,
            confidence=qwen_fields.confidence or classic_fields.confidence,
        )


class PaddleOcrVlExtractor:
    def __init__(self, pipeline_version: str = "v1.6", engine: str | None = None) -> None:
        self.pipeline_version = pipeline_version
        self.engine = engine
        self._pipeline = None

    def extract(self, image_path: Path) -> ExtractedFields:
        text = self._run_paddleocr(image_path)
        return ExtractedFields(
            name=extract_name_from_text(text),
            national_id=extract_national_id(text),
            ocr_text=text,
        )

    def _run_paddleocr(self, image_path: Path) -> str:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL is not installed. Install paddlepaddle and "
                "'paddleocr[doc-parser]>=3.6.0' in the runtime environment."
            ) from exc

        if self._pipeline is None:
            kwargs = {"pipeline_version": self.pipeline_version}
            if self.engine:
                kwargs["engine"] = self.engine
            self._pipeline = PaddleOCRVL(**kwargs)

        results = self._pipeline.predict(str(image_path))
        return flatten_paddleocr_results(results)


def flatten_paddleocr_results(results: object) -> str:
    chunks: list[str] = []

    def walk(value: object) -> None:
        if value is None:
            return
        if isinstance(value, str):
            chunks.append(value)
            return
        if isinstance(value, dict):
            for key in ("text", "rec_text", "rec_texts", "content", "markdown", "html"):
                item = value.get(key)
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, (dict, list, tuple)):
                    walk(item)
            for item in value.values():
                if isinstance(item, (dict, list, tuple)):
                    walk(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
            return
        json_value = getattr(value, "json", None)
        if isinstance(json_value, dict):
            walk(json_value)
            return
        if callable(json_value):
            try:
                walk(json_value())
                return
            except TypeError:
                pass
        if hasattr(value, "__dict__"):
            walk(vars(value))

    walk(results)
    return "\n".join(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))

