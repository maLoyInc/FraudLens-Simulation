"""The feature contract: names, order, count, and the two approved changes."""

from __future__ import annotations

import json

from fraudlens.core import config as cfg
from fraudlens.core import features as spec


def test_spec_is_self_consistent():
    # Raises if the lists ever drift apart.
    spec.assert_feature_spec_consistent()


def test_sixteen_features_in_a_fixed_order():
    assert len(spec.FEATURE_ORDER) == 16
    assert len(set(spec.FEATURE_ORDER)) == 16


def test_categorical_and_numeric_partition_the_feature_set():
    assert set(spec.CATEGORICAL_FEATURES) | set(spec.NUMERIC_FEATURES) == set(
        spec.FEATURE_ORDER
    )
    assert not set(spec.CATEGORICAL_FEATURES) & set(spec.NUMERIC_FEATURES)


def test_city_pop_is_removed_not_hidden():
    assert "city_pop" not in spec.FEATURE_ORDER
    assert "city_pop" not in spec.CATEGORICAL_FEATURES
    assert "city_pop" not in spec.NUMERIC_FEATURES
    assert "city_pop" in spec.DROPPED_RAW_COLUMNS


def test_transaction_seconds_is_a_real_feature():
    assert "transaction_seconds" in spec.FEATURE_ORDER
    assert "transaction_seconds" in spec.NUMERIC_FEATURES


def test_month_maps_to_ordered_integers():
    assert len(spec.MONTH_NAMES) == 12
    assert spec.MONTH_NAME_TO_NUMBER["January"] == 1
    assert spec.MONTH_NAME_TO_NUMBER["December"] == 12


def test_weekday_names_are_the_pandas_day_names():
    assert spec.WEEKDAY_NAMES[0] == "Monday"
    assert spec.WEEKDAY_NAMES[-1] == "Sunday"
    assert len(spec.WEEKDAY_NAMES) == 7


def test_category_normalisation_matches_the_processed_build():
    assert spec.normalise_category("grocery_pos") == "Grocery Pos"
    assert spec.normalise_category("gas_transport") == "Gas Transport"


def test_saved_metadata_agrees_with_the_spec():
    metadata = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["feature_order"] == spec.FEATURE_ORDER
    assert metadata["n_features"] == len(spec.FEATURE_ORDER)
    assert metadata["categorical_features"] == spec.CATEGORICAL_FEATURES
    assert metadata["numeric_features"] == spec.NUMERIC_FEATURES
    assert metadata["target"] == spec.TARGET


def test_input_ranges_match_the_product_rules():
    assert (spec.AGE_MIN, spec.AGE_MAX) == (1, 120)
    assert (spec.HOUR_MIN, spec.HOUR_MAX) == (0, 23)
    assert (spec.MINUTE_MIN, spec.MINUTE_MAX) == (0, 59)
    assert (spec.SECOND_MIN, spec.SECOND_MAX) == (0, 59)
    assert (spec.DATE_MIN, spec.DATE_MAX) == (1, 31)
