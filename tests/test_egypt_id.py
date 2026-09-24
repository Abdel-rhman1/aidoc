from aidoc.egypt_id import extract_national_id, validate_egyptian_national_id


def test_extracts_arabic_digits_id() -> None:
    assert extract_national_id("الرقم القومي ٢٩٩٠١٠١١٢٣٤٥٦٧") == "29901011234578"


def test_valid_id_structure() -> None:
    result = validate_egyptian_national_id("29901011234578")
    assert result.valid
    assert result.birth_date == "1999-01-01"
    assert result.governorate == "Dakahlia"
    assert result.gender == "male"


def test_invalid_date() -> None:
    result = validate_egyptian_national_id("29902301234567")
    assert not result.valid
    assert "invalid_birth_date" in result.errors



def test_extracts_spaced_arabic_digits_id() -> None:
    assert extract_national_id("٢ ٨٥ ٠٦ ٠٦ ١٢ ٠٠٤٧٥") == "28506061200475"


def test_extracts_id_from_reversed_ocr_chunks() -> None:
    assert extract_national_id("٦٠٦١٢٠٠٤٧٥\n٢٨٥") == "28506061200475"
