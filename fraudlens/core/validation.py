"""Input parsing and step validation.

Framework-free so it can be unit tested without Streamlit. Each validator
returns a list of human-readable messages; an empty list means the step is
complete and safe to advance.
"""

from __future__ import annotations

import re

from . import features as spec
from . import reference as ref

_AMOUNT_ALLOWED = re.compile(r"^[0-9,. ]+$")


def parse_amount(raw: str | None) -> tuple[float | None, str | None]:
    """Parse a typed amount into a float.

    Accepts the display form the UI writes back (``34,234.00``) as well as a
    plain ``34234`` or ``34234.5``. Returns ``(value, error)``; exactly one of
    the two is ``None``.
    """
    if raw is None:
        return None, None
    text = raw.strip().replace("$", "").strip()
    if not text:
        return None, None
    if not _AMOUNT_ALLOWED.match(text):
        return None, "Amount may only contain digits, a decimal point and thousands separators."

    text = text.replace(" ", "").replace(",", "")
    if text.count(".") > 1:
        return None, "Amount has more than one decimal point."
    try:
        value = float(text)
    except ValueError:
        return None, "Amount is not a valid number."

    if value < spec.AMOUNT_MIN:
        return None, f"Amount must be at least ${spec.AMOUNT_MIN:.2f}."
    return round(value, 2), None


def format_amount(value: float) -> str:
    """Float -> the display form used in the field and the review summary."""
    return f"{value:,.2f}"


def clamp_age(value: int | float | None) -> int | None:
    """Constrain an age to the product range instead of rejecting it.

    PRD section 6.1 asks for values above 120 to be sanitised to 120 rather
    than treated as invalid.
    """
    if value is None:
        return None
    return int(min(max(int(value), spec.AGE_MIN), spec.AGE_MAX))


# ---------------------------------------------------------------------------
# Step validators
# ---------------------------------------------------------------------------
def validate_customer(gender, age, job) -> list[str]:
    problems: list[str] = []
    if not gender:
        problems.append("Gender is required.")
    if age is None:
        problems.append("Age is required.")
    elif not (spec.AGE_MIN <= age <= spec.AGE_MAX):
        problems.append(f"Age must be between {spec.AGE_MIN} and {spec.AGE_MAX}.")
    if not job:
        problems.append("Job is required.")
    return problems


def validate_location(state, city, street, zip_code) -> list[str]:
    problems: list[str] = []
    if not state:
        problems.append("State is required.")
        return problems
    if not city:
        problems.append("City is required.")
    elif city not in ref.cities(state):
        problems.append(f"{city} is not a city recorded in {state}.")
    if not city:
        return problems

    if not street:
        problems.append("Street is required.")
    elif street not in ref.streets(state, city):
        problems.append(f"{street} is not a street recorded in {city}, {state}.")
    if not street:
        return problems

    if not zip_code:
        problems.append("ZIP Code is required.")
    elif zip_code not in ref.zips(state, city, street):
        problems.append("The selected ZIP Code is not valid for this address.")
    elif ref.resolve_coordinates(state, city, street, zip_code) is None:
        problems.append("Customer coordinates could not be resolved for this address.")
    return problems


def validate_transaction(
    amount_raw, category, day_of_week, month_name, date, hour, minute, second
) -> list[str]:
    problems: list[str] = []

    amount, amount_error = parse_amount(amount_raw)
    if amount_error:
        problems.append(amount_error)
    elif amount is None:
        problems.append("Amount is required.")

    if not category:
        problems.append("Category is required.")
    if not day_of_week:
        problems.append("Week is required.")
    if not month_name:
        problems.append("Month is required.")

    if date is None:
        problems.append("Date is required.")
    elif not (spec.DATE_MIN <= int(date) <= spec.DATE_MAX):
        problems.append(f"Date must be between {spec.DATE_MIN} and {spec.DATE_MAX}.")

    for value, name, low, high in (
        (hour, "Hour", spec.HOUR_MIN, spec.HOUR_MAX),
        (minute, "Minute", spec.MINUTE_MIN, spec.MINUTE_MAX),
        (second, "Second", spec.SECOND_MIN, spec.SECOND_MAX),
    ):
        if value is None:
            problems.append(f"{name} is required.")
        elif not (low <= int(value) <= high):
            problems.append(f"{name} must be between {low:02d} and {high:02d}.")

    return problems
