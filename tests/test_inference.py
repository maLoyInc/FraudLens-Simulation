"""Inference: artifact loading, row construction, and prediction behaviour."""

from __future__ import annotations

import dataclasses

import pytest

from fraudlens.core import features as spec
from fraudlens.core import reference as ref
from fraudlens.inference import predictor as inf


def sample_input(**overrides) -> inf.TransactionInput:
    state = ref.states()[0]
    city = ref.cities(state)[0]
    street = ref.streets(state, city)[0]
    zip_code = ref.zips(state, city, street)[0]
    lat, long = ref.resolve_coordinates(state, city, street, zip_code)
    base = inf.TransactionInput(
        gender=ref.genders()[0], age=42, job=ref.jobs()[0],
        state=state, city=city, street=street, zip_code=zip_code,
        lat=lat, long=long,
        amount=120.50, category=ref.categories()[0], day_of_week="Monday",
        month_name="March", date=14, hour=22, minute=5, second=9,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def test_artifacts_load_and_are_cached():
    first = inf.load_artifacts()
    second = inf.load_artifacts()
    assert first is second
    encoder, scaler, model, metadata = first
    assert metadata["feature_order"] == spec.FEATURE_ORDER
    assert scaler.n_features_in_ == len(spec.FEATURE_ORDER)
    assert len(encoder.categories_) == len(spec.CATEGORICAL_FEATURES)
    assert model.n_features_in_ == len(spec.FEATURE_ORDER)


def test_model_row_has_exactly_the_model_features_in_order():
    row, _, _, _ = inf.build_model_row(sample_input())
    assert list(row.keys()) == spec.FEATURE_ORDER


def test_model_row_translates_the_presentation_values():
    row, _, _, _ = inf.build_model_row(sample_input(month_name="December"))
    assert row["transaction_month"] == 12
    assert row["day_of_week"] == "Monday"
    assert row["transaction_seconds"] == 9


def test_model_row_is_deterministic_for_identical_input():
    a = inf.build_model_row(sample_input())
    b = inf.build_model_row(sample_input())
    assert a == b


def test_changing_a_field_changes_the_derived_distance():
    _, first, _, _ = inf.build_model_row(sample_input())
    _, second, _, _ = inf.build_model_row(sample_input(second=10))
    assert first != second


def test_prediction_shape_and_bounds():
    prediction = inf.predict(sample_input())
    assert isinstance(prediction.is_fraud, bool)
    assert 0.0 <= prediction.fraud_probability <= 1.0
    assert prediction.transaction_distance_km > 0
    assert list(prediction.model_row.keys()) == spec.FEATURE_ORDER


def test_prediction_agrees_with_the_documented_decision_rule():
    prediction = inf.predict(sample_input())
    assert prediction.is_fraud == (prediction.fraud_probability >= 0.50)


def test_prediction_is_reproducible():
    a = inf.predict(sample_input())
    b = inf.predict(sample_input())
    assert a.is_fraud == b.is_fraud
    assert a.fraud_probability == b.fraud_probability
    assert a.transaction_distance_km == b.transaction_distance_km


def test_unseen_categorical_value_does_not_crash_inference():
    # The encoder was fitted with unknown_value=-1, so an out-of-vocabulary
    # string must still score rather than raise.
    prediction = inf.predict(sample_input(job="Interdimensional Plumber"))
    assert 0.0 <= prediction.fraud_probability <= 1.0


def test_missing_month_name_is_rejected_loudly():
    with pytest.raises(KeyError):
        inf.build_model_row(sample_input(month_name="Smarch"))


def test_transaction_input_is_immutable():
    tx = sample_input()
    with pytest.raises(dataclasses.FrozenInstanceError):
        tx.amount = 1.0
