from aidoc.extractor import ExtractedFields, HybridExtractor, extract_name_from_text, json_block_from_text


def test_extracts_name_when_first_name_is_separate_line() -> None:
    text = """احمد
محمد سمير احمد
٣ ش خليل
"""

    assert extract_name_from_text(text) == "أحمد محمد سمير"


def test_skips_noisy_republic_header_when_text_is_collapsed() -> None:
    text = (
        "حمهَريَ مصَ الحَربيَُة الشخصية بطاقة تحقيق "
        "احمد محمد احمد محمد سند ٣ ش عبدالحليم منزل القفاش رقم ٣ "
        "اول المنتزه الاسكندرية ١٩٨٠١٠٩/٢٧ ٢٨٠٠٩٢٧٠٢٠٢٩٣٧"
    )

    assert extract_name_from_text(text) == "أحمد محمد احمد محمد سند"


def test_skips_noisy_republic_header_line() -> None:
    text = """حمهَريَ مصَ الحَربيَُة
الشخصية
بطاقة
تحقيق
احمد محمد احمد محمد سند
٣ ش عبدالحليم
"""

    assert extract_name_from_text(text) == "أحمد محمد احمد محمد سند"


def test_parses_qwen_json_block() -> None:
    data = json_block_from_text(
        """```json
{"name": "محمد عبده سالم عميره", "national_id_candidates": ["KW4749408"], "address": "اسكندرية"}
```"""
    )

    assert data["name"] == "محمد عبده سالم عميره"
    assert data["national_id_candidates"] == ["KW4749408"]


class FakeExtractor:
    def __init__(self, fields: ExtractedFields) -> None:
        self.fields = fields
        self.calls = 0

    def extract(self, _image_path):
        self.calls += 1
        return self.fields


def test_hybrid_runs_qwen_when_classic_has_no_id(tmp_path) -> None:
    classic = FakeExtractor(ExtractedFields("محمد عبده سالم", None, "KW4749408"))
    qwen = FakeExtractor(
        ExtractedFields(
            "محمد عبده سالم عميره",
            "KW4749408",
            "٢٨٠٠٩٢٧٠٢٠٢٩٣٧",
        )
    )
    hybrid = HybridExtractor(classic=classic, qwen=qwen)

    fields = hybrid.extract(tmp_path / "id.jpg")

    assert classic.calls == 1
    assert qwen.calls == 1
    assert fields.name == "محمد عبده سالم عميره"
    assert fields.national_id == "28009270202937"


def test_hybrid_skips_qwen_when_classic_has_valid_id(tmp_path) -> None:
    classic = FakeExtractor(ExtractedFields("محمد عبده سالم", "28009270202937", ""))
    qwen = FakeExtractor(ExtractedFields("wrong", None, ""))
    hybrid = HybridExtractor(classic=classic, qwen=qwen)

    fields = hybrid.extract(tmp_path / "id.jpg")

    assert classic.calls == 1
    assert qwen.calls == 0
    assert fields.national_id == "28009270202937"
