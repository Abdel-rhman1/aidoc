from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from aidoc.api import app
from aidoc.extractor import ExtractedFields


def make_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_rejects_non_image_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/verify",
        files={"file": ("id.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_invalid_image_bytes() -> None:
    client = TestClient(app)

    response = client.post(
        "/verify",
        files={"file": ("id.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400


def test_verifies_valid_uploaded_image(monkeypatch) -> None:
    def fake_extract(_path):
        return ExtractedFields(
            name="احمد محمد علي",
            national_id="29901011234578",
            ocr_text="الرقم القومي ٢٩٩٠١٠١١٢٣٤٥٦٧",
        )

    monkeypatch.setattr("aidoc.api.extractor.extract", fake_extract)
    client = TestClient(app)

    response = client.post(
        "/verify",
        files={"file": ("id.jpg", make_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["valid"] is True
