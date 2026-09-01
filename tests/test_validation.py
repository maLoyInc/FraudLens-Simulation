"""Amount parsing, age clamping and the per-step validators."""

from __future__ import annotations

import pytest

from fraudlens.core import features as spec
from fraudlens.core import reference as ref
from fraudlens.core import validation as val


@pytest.mark.parametrize("raw,expected", [
    ("34234", 34234.00),
    ("34234.5", 34234.50),
    ("34,234.00", 34234.00),
    ("34 234.00", 34234.00),
    ("$ 34,234.00", 34234.00),
    ("0.01", 0.01),
    ("1.005", 1.00),
])
def test_amount_accepts_the_display_and_plain_forms(raw, expected):
    value, error = val.parse_amount(raw)
    assert error is None
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_blank_amount_is_empty_not_invalid(raw):
    value, error = val.parse_amount(raw)
    assert value is None and error is None


@pytest.mark.parametrize("raw", ["abc", "12a", "-5", "1e5", "12/3"])
def test_amount_rejects_non_numeric_text(raw):
    value, error = val.parse_amount(raw)
    assert value is None and error


def test_amount_rejects_two_decimal_points():
    value, error = val.parse_amount("1.2.3")
    assert value is None and "decimal point" in error


def test_amount_rejects_below_the_minimum():
    value, error = val.parse_amount("0")
    assert value is None and error


def test_format_amount_is_the_display_form():
    assert val.format_amount(34234.0) == "34,234.00"
    assert val.format_amount(7.5) == "7.50"


@pytest.mark.parametrize("raw,expected", [
    (150, spec.AGE_MAX), (121, spec.AGE_MAX), (120, 120),
    (0, spec.AGE_MIN), (-4, spec.AGE_MIN), (42, 42), (42.9, 42), (None, None),
])
def test_age_is_clamped_rather_than_rejected(raw, expected):
    assert val.clamp_age(raw) == expected


def test_customer_step_lists_every_missing_field():
    problems = val.validate_customer(None, None, None)
    assert len(problems) == 3


def test_customer_step_passes_when_complete():
    assert val.validate_customer("F", 42, ref.jobs()[0]) == []


def test_location_step_stops_at_the_first_missing_level():
    assert val.validate_location(None, None, None, None) == ["State is required."]


def test_location_step_rejects_a_stale_downstream_value():
    state = ref.states()[0]
    problems = val.validate_location(state, "Nowhere City", None, None)
    assert problems and "Nowhere City" in problems[0]


def test_location_step_passes_for_a_real_address():
    state = ref.states()[0]
    city = ref.cities(state)[0]
    street = ref.streets(state, city)[0]
    zip_code = ref.zips(state, city, street)[0]
    assert val.validate_location(state, city, street, zip_code) == []


def test_transaction_step_lists_every_missing_field():
    problems = val.validate_transaction(None, None, None, None, None, None, None, None)
    assert len(problems) == 8


def test_transaction_step_flags_out_of_range_time():
    problems = val.validate_transaction(
        "10.00", "Grocery Pos", "Monday", "March", 14, 24, 60, 60
    )
    assert len(problems) == 3


def test_transaction_step_passes_when_complete():
    problems = val.validate_transaction(
        "34,234.00", "Grocery Pos", "Monday", "March", 14, 22, 5, 9
    )
    assert problems == []
