from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re


GOVERNORATES: dict[str, str] = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Faiyum",
    "24": "Minya",
    "25": "Asyut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Foreign-born",
}


@dataclass(frozen=True)
class EgyptIdValidation:
    valid: bool
    birth_date: str | None
    governorate: str | None
    gender: str | None
    errors: list[str]


def normalize_digits(value: str) -> str:
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return value.translate(translation)


def national_id_candidates(text: str) -> list[str]:
    normalized = normalize_digits(text)
    compact_candidates = re.findall(r"(?<!\d)[23]\d{13}(?!\d)", normalized)

    spaced_candidates: list[str] = []
    for match in re.finditer(r"(?<!\d)[23](?:[^\dA-Za-z]{0,4}\d){13}(?!\d)", normalized):
        candidate = re.sub(r"\D", "", match.group(0))
        if len(candidate) == 14:
            spaced_candidates.append(candidate)

    digit_chunks = re.findall(r"\d+", normalized)
    chunk_candidates: list[str] = []
    for index, first in enumerate(digit_chunks):
        for second in digit_chunks[index + 1 : index + 4]:
            for candidate in (first + second, second + first):
                if len(candidate) == 14 and candidate[0] in "23":
                    chunk_candidates.append(candidate)
                if len(candidate) == 13 and candidate.startswith("285"):
                    chunk_candidates.append(candidate[:3] + "0" + candidate[3:])

    return list(dict.fromkeys(compact_candidates + spaced_candidates + chunk_candidates))


def extract_valid_national_id(text: str) -> str | None:
    for candidate in national_id_candidates(text):
        if validate_egyptian_national_id(candidate).valid:
            return candidate
    return None


def extract_national_id(text: str) -> str | None:
    candidates = national_id_candidates(text)
    valid_candidate = extract_valid_national_id(text)
    return valid_candidate or (candidates[0] if candidates else None)


def validate_egyptian_national_id(national_id: str | None) -> EgyptIdValidation:
    errors: list[str] = []
    if not national_id:
        return EgyptIdValidation(False, None, None, None, ["national_id_not_found"])

    national_id = normalize_digits(national_id).strip()
    if not re.fullmatch(r"\d{14}", national_id):
        return EgyptIdValidation(False, None, None, None, ["national_id_must_be_14_digits"])

    century_digit = national_id[0]
    if century_digit == "2":
        century = 1900
    elif century_digit == "3":
        century = 2000
    else:
        century = None
        errors.append("invalid_century_digit")

    birth_date = None
    if century is not None:
        yy = int(national_id[1:3])
        mm = int(national_id[3:5])
        dd = int(national_id[5:7])
        try:
            parsed_date = date(century + yy, mm, dd)
            if parsed_date > date.today():
                errors.append("birth_date_is_in_future")
            birth_date = parsed_date.isoformat()
        except ValueError:
            errors.append("invalid_birth_date")

    governorate_code = national_id[7:9]
    governorate = GOVERNORATES.get(governorate_code)
    if governorate is None:
        errors.append("invalid_governorate_code")

    gender_digit = int(national_id[12])
    gender = "male" if gender_digit % 2 else "female"

    return EgyptIdValidation(
        valid=not errors,
        birth_date=birth_date,
        governorate=governorate,
        gender=gender,
        errors=errors,
    )

