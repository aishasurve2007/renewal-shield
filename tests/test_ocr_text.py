"""Text side of the OCR pipeline (no OCR engine needed): clause splitting, field parsing, scoring."""
import pytest

from check_contract import check_fields
from ocr_pipeline import _date, _num, extract_fields, match_clauses, split_clauses

C1 = "Any increase in rent upon renewal shall be in accordance with the rental index issued by the Dubai Land Department."
C2 = "The security deposit shall be refunded to the Tenant within thirty (30) days of vacating the premises."
C3 = "The Tenant shall repair any damage caused by his own misuse or negligence during the term."

FIELDS = """TENANCY CONTRACT
Contract Date: 01/07/2026   Unit No: 1204
Start Date: 15/07/2026   End Date: 14/07/2027
Annual Rent: AED 85,000   Contract Value: AED 85,000
Security Deposit: AED 4,250   Payment Mode: Cheque
No. of Cheques: 4
"""


def contract(body, heading="4. ADDITIONAL TERMS"):
    return f"{FIELDS}{heading}\n{body}\nLandlord Signature            Tenant Signature\n"


# ----------------------------------------------------------------- clause splitting
def test_split_numbered_clauses():
    assert split_clauses(contract(f"1. {C1}\n2. {C2}\n3. {C3}")) == [C1, C2, C3]


def test_split_joins_wrapped_lines():
    half = C1.index("in accordance")
    text = contract(f"1. {C1[:half]}\n{C1[half:]}\n2. {C2}")
    assert split_clauses(text) == [C1, C2]


@pytest.mark.parametrize("n2", ["2..", "2:", "2,", "2.)"])
def test_split_accepts_clause_numbers_as_ocr_reads_them(n2):
    assert split_clauses(contract(f"1. {C1}\n{n2} {C2}")) == [C1, C2]


@pytest.mark.parametrize("heading", ["4. ADDITIONAL.TERMS", "4. ADD1TIONAL TERMS", "4. ADDlTIONAL TERMS"])
def test_split_tolerates_a_misread_heading(heading):
    assert split_clauses(contract(f"1. {C1}\n2. {C2}", heading)) == [C1, C2]


def test_field_lines_never_become_clauses():
    got = split_clauses(contract(f"1. {C1}"))
    assert got == [C1] and not any("Annual Rent" in c or "Cheques" in c for c in got)


@pytest.mark.parametrize("sig", ["Landlord Signature     Tenant Signature", "Landlurd Sigaatury", "Laodlord Signabure"])
def test_split_stops_at_a_mangled_signature_line(sig):
    text = f"{FIELDS}4. ADDITIONAL TERMS\n1. {C1}\n2. {C2}\n{sig}\n"
    assert split_clauses(text) == [C1, C2]


def test_split_repairs_lf_for_if():
    clause = "lf the Tenant finds a replacement tenant acceptable to the Landlord, no penalty shall apply."
    assert split_clauses(contract(f"1. {C1}\n2.{clause}"))[1].startswith("If the Tenant")


def test_split_text_without_any_contract_layout_returns_something_safe():
    assert split_clauses("") == []


# ----------------------------------------------------------------- fields
@pytest.mark.parametrize("raw,want", [
    ("15/07/2027", "2027-07-15"),
    ("15/072027", "2027-07-15"),          # OCR dropped a slash
    ("1105/2026", "2026-05-11"),
    ("5 / 1 / 2026", "2026-01-05"),
    ("32/01/2026", None),                 # impossible day
    ("15/13/2026", None),                 # impossible month
    ("garbage", None),
    ("", None),
])
def test_date_parsing(raw, want):
    assert _date(raw) == want


@pytest.mark.parametrize("raw,want", [("85,000", 85000), ("47,ooo", 47000), ("4,25O", 4250), ("12 500", 12500), ("", None)])
def test_amount_parsing_tolerates_ocr_zeros(raw, want):
    assert _num(raw) == want


def test_extract_fields_from_a_clean_form():
    f = extract_fields(FIELDS)
    assert f == {"contract_date": "2026-07-01", "start_date": "2026-07-15", "end_date": "2027-07-14",
                 "annual_rent": 85000, "contract_value": 85000, "security_deposit": 4250,
                 "num_cheques": 4, "payment_mode": "Cheque", "unit_no": "1204"}


def test_extract_fields_from_noisy_ocr():
    noisy = (FIELDS.replace("AED 85,000   Contract", "AFD 85,ooo   Contract")
             .replace("Contract Date", "Contrtest Date").replace("14/07/2027", "14/072027"))
    f = extract_fields(noisy)
    assert f["annual_rent"] == 85000 and f["contract_date"] == "2026-07-01" and f["end_date"] == "2027-07-14"


def test_missing_fields_are_none_not_errors():
    f = extract_fields("nothing useful here")
    assert set(f.values()) == {None}


# ----------------------------------------------------------------- scoring
def test_match_clauses_does_not_cascade_when_one_clause_is_missing():
    """The old in-order scorer let one missing clause shift every later match (RESULTS.md 3)."""
    pairs = match_clauses([C1, C3], [C1, C2, C3])
    assert [round(sim, 2) for _, _, sim in pairs][0] == 1.0 and pairs[2][1] == C3
    assert pairs[1][2] < 0.6


# ----------------------------------------------------------------- field checks
def test_check_fields_is_quiet_on_a_normal_contract():
    assert check_fields(extract_fields(FIELDS)) == []


def test_check_fields_flags_inconsistent_numbers():
    f = extract_fields(FIELDS)
    f.update(security_deposit=20000, end_date="2026-01-01", payment_mode="Bank Transfer")
    notes = " ".join(check_fields(f))
    assert "Security deposit" in notes and "End date is before" in notes and "bank transfer" in notes
