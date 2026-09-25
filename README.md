# Egyptian National ID OCR

FastAPI + Streamlit proof of concept for reading the front side of an Egyptian national ID card with PaddleOCR-VL 1.6, extracting the holder name and national ID number, and validating the ID structure.

## What It Verifies

This project validates the 14-digit Egyptian National ID format:

- century digit: `2` for 1900-1999, `3` for 2000-2099
- birth date: valid `YYMMDD`
- governorate code: known Egyptian governorate or `88` for foreign-born
- serial/gender digit: odd means male, even means female

It does **not** verify the ID against a government database. Treat the result as document OCR + structural validation.

## Recommended Architecture

Run PaddleOCR-VL as the OCR engine and keep the API/UI lightweight:

```text
image upload
   -> FastAPI /verify
   -> PaddleOCR-VL OCR/parser
   -> field extraction
   -> Egyptian NID validator
   -> JSON response
   -> Streamlit UI
```

For production, run PaddleOCR-VL in a separate GPU service. PaddleOCR's current docs recommend Docker or a dedicated VLM inference service for production stability. Local Python loading is fine for development, but startup and inference are heavy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PaddleOCR-VL runtime separately according to your machine:

```bash
# CPU example
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
```

For NVIDIA GPU, install the matching `paddlepaddle-gpu` wheel instead.

## Run

API:

```bash
uvicorn src.aidoc.api:app --reload --host 0.0.0.0 --port 8000
```

By default the API uses the lighter Arabic PaddleOCR engine so it can run on lower-memory devices. PaddleOCR-VL is much heavier and may fill RAM or get killed by the OS on normal devices. Use it only on a strong machine or GPU server:

```bash
AIDOC_OCR_ENGINE=vl uvicorn src.aidoc.api:app --host 0.0.0.0 --port 8000
```

To test a Qwen OpenVINO VLM on the server, install OpenVINO dependencies in that runtime and point the API at the downloaded model directory:

```bash
AIDOC_OCR_ENGINE=qwen \
QWEN_VL_MODEL_PATH=/root/models/Qwen3-VL-4B-Instruct-int4-ov \
QWEN_VL_DEVICE=CPU \
uvicorn src.aidoc.api:app --host 0.0.0.0 --port 8000
```

The API still validates the national ID structurally, so VLM serial/code outputs such as `KW4749408` are rejected.

Hybrid mode runs the lighter classic OCR first, then falls back to Qwen if the classic result is missing a valid ID or name:

```bash
AIDOC_OCR_ENGINE=hybrid \
QWEN_VL_MODEL_PATH=/root/models/Qwen3-VL-4B-Instruct-int4-ov \
QWEN_VL_DEVICE=CPU \
uvicorn src.aidoc.api:app --host 0.0.0.0 --port 8000
```

Set `HYBRID_ALWAYS_QWEN=1` to always run both engines for comparison.

UI:

```bash
streamlit run streamlit_app.py
```

## API

```bash
curl -X POST http://localhost:8000/verify \
  -F "file=@/path/to/egyptian-id-front.jpg"
```

Example response:

```json
{
  "valid": true,
  "name": "احمد محمد علي",
  "national_id": "29901011234578",
  "validation": {
    "valid": true,
    "birth_date": "1999-01-01",
    "governorate": "Dakahlia",
    "gender": "male",
    "errors": []
  },
  "ocr_text": "...",
  "confidence": null
}
```

