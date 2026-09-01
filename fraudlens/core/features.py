"""The FraudLens v2 feature specification.

This module is the single source of truth for what the model consumes and in
which order. Training builds its matrix from :data:`FEATURE_ORDER` and inference
builds its row from the same list, so the two cannot drift apart.

Changes relative to the legacy 16-feature set
---------------------------------------------
removed  ``city_pop``            - required by PRD section 8 / CLAUDE.md section 19
added    ``transaction_seconds`` - required by PRD section 10.6 / CLAUDE.md section 20

The count is still 16 because one feature was removed and one added.

Representation decisions
------------------------
``transaction_month``
    Kept as the integer 1-12 used by the research pipeline. Month *names* are
    only a presentation layer (see :data:`MONTH_NAMES`). Encoding names through
    an ordinal encoder would sort them alphabetically (April=0, August=1, ...)
    and destroy the natural ordering, so the numeric form is the better
    preprocessing choice and the UI maps between the two.

``day_of_week``
    Kept as the English day name string the research pipeline produced, and
    ordinal encoded. It is nominal, so alphabetical codes are not a problem.

``category``
    Kept in the research pipeline's normalised form (``grocery_pos`` ->
    ``Grocery Pos``). :data:`CATEGORY_DISPLAY` supplies friendlier UI wording
    without changing the value handed to the encoder.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "category",
    "gender",
    "state",
    "city",
    "street",
    "job",
    "day_of_week",
]

NUMERIC_FEATURES: list[str] = [
    "amt",
    "zip",
    "age",
    "transaction_hour",
    "transaction_min",
    "transaction_seconds",
    "transaction_date",
    "transaction_month",
    "transaction_distance",
]

# Explicit model input order. Inference must build columns in exactly this
# order; the order is also written into models/v2/model_metadata.json.
FEATURE_ORDER: list[str] = [
    "category",
    "amt",
    "gender",
    "state",
    "city",
    "street",
    "zip",
    "job",
    "age",
    "day_of_week",
    "transaction_hour",
    "transaction_min",
    "transaction_seconds",
    "transaction_date",
    "transaction_month",
    "transaction_distance",
]

TARGET = "is_fraud"

# Legacy feature set, kept only for the comparison written into the metadata.
LEGACY_FEATURE_ORDER: list[str] = [
    "category", "amt", "gender", "state", "city", "street", "zip",
    "city_pop", "job", "age", "day_of_week", "transaction_min",
    "transaction_hour", "transaction_date", "transaction_month",
    "transaction_distance",
]

# Raw columns dropped before modelling, with the reason.
DROPPED_RAW_COLUMNS: dict[str, str] = {
    "Unnamed: 0": "row index artefact of the CSV export",
    "unix_time": "redundant with trans_date_trans_time",
    "cc_num": "customer identifier, not a behavioural feature; also sensitive",
    "first": "personal name, not a feature",
    "last": "personal name, not a feature",
    "merchant": "not a model feature in the research pipeline; kept out for continuity",
    "trans_num": "unique transaction id, no predictive content",
    "dob": "consumed by the derived age feature",
    "trans_date_trans_time": "consumed by the derived time features",
    "lat": "consumed by transaction_distance",
    "long": "consumed by transaction_distance",
    "merch_lat": "consumed by transaction_distance",
    "merch_long": "consumed by transaction_distance",
    "city_pop": "removed from the final model by product decision (PRD section 8)",
}

# ---------------------------------------------------------------------------
# Presentation maps (UI only; never sent to the model)
# ---------------------------------------------------------------------------
MONTH_NAMES: list[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_NAME_TO_NUMBER: dict[str, int] = {n: i + 1 for i, n in enumerate(MONTH_NAMES)}

WEEKDAY_NAMES: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

GENDER_DISPLAY: dict[str, str] = {"F": "Female", "M": "Male"}

# Model value -> label shown in the form. Keys are the normalised values the
# encoder was fitted on.
CATEGORY_DISPLAY: dict[str, str] = {
    "Entertainment": "Entertainment",
    "Food Dining": "Food & Dining",
    "Gas Transport": "Gas & Transport",
    "Grocery Net": "Grocery (online)",
    "Grocery Pos": "Grocery (in store)",
    "Health Fitness": "Health & Fitness",
    "Home": "Home",
    "Kids Pets": "Kids & Pets",
    "Misc Net": "Miscellaneous (online)",
    "Misc Pos": "Miscellaneous (in store)",
    "Personal Care": "Personal Care",
    "Shopping Net": "Shopping (online)",
    "Shopping Pos": "Shopping (in store)",
    "Travel": "Travel",
}

FEATURE_LABELS: dict[str, str] = {
    "category": "Category",
    "amt": "Amount",
    "gender": "Gender",
    "state": "State",
    "city": "City",
    "street": "Street",
    "zip": "ZIP Code",
    "job": "Job",
    "age": "Age",
    "day_of_week": "Week",
    "transaction_hour": "Hour",
    "transaction_min": "Minute",
    "transaction_seconds": "Second",
    "transaction_date": "Date",
    "transaction_month": "Month",
    "transaction_distance": "Transaction Distance",
}

# Validation bounds. Calendar-derived where a calendar rule exists, otherwise
# from the raw audit (training/reports/raw_audit.json).
AGE_MIN, AGE_MAX = 1, 120           # product rule, PRD section 6.1
AMOUNT_MIN = 0.01                   # a transaction amount must be positive
HOUR_MIN, HOUR_MAX = 0, 23
MINUTE_MIN, MINUTE_MAX = 0, 59
SECOND_MIN, SECOND_MAX = 0, 59
DATE_MIN, DATE_MAX = 1, 31


def normalise_category(raw: str) -> str:
    """Raw ``category`` value -> the form the model was trained on."""
    return raw.replace("_", " ").title()


def assert_feature_spec_consistent() -> None:
    """Fail loudly if the three feature lists ever disagree."""
    combined = set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
    ordered = set(FEATURE_ORDER)
    if combined != ordered:
        missing = sorted(ordered - combined)
        extra = sorted(combined - ordered)
        raise AssertionError(
            f"feature spec mismatch: not typed={missing} not ordered={extra}"
        )
    if len(FEATURE_ORDER) != len(set(FEATURE_ORDER)):
        raise AssertionError("FEATURE_ORDER contains duplicates")
    if "city_pop" in ordered:
        raise AssertionError("city_pop must not be a v2 feature")
    if "transaction_seconds" not in ordered:
        raise AssertionError("transaction_seconds must be a v2 feature")


assert_feature_spec_consistent()
