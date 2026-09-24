from aidoc.extractor import extract_name_from_text


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
