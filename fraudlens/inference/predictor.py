"""Inference-only prediction pipeline.

Loads the artifacts produced by ``training/scripts/03_train_xgboost.py`` and
turns validated form input into a prediction. This module never calls ``fit``,
never pickles anything, and never opens a file for writing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import pandas as pd

from ..core import config as cfg
from ..core import features as spec
from ..core.geo import resolve_transaction_distance


@dataclass(frozen=True)
class TransactionInput:
    """One fully specified transaction, in UI terms."""

    # Customer
    gender: str
    age: int
    job: str
    # Location
    state: str
    city: str
    street: str
    zip_code: str
    lat: float
    long: float
    # Transaction
    amount: float
    category: str
    day_of_week: str
    month_name: str
    date: int
    hour: int
    minute: int
    second: int


@dataclass(frozen=True)
class Prediction:
    is_fraud: bool
    fraud_probability: float
    transaction_distance_km: float
    merchant_lat: float
    merchant_long: float
    model_row: dict = field(repr=False)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load encoder, scaler, model and metadata. Read-only."""
    for path in (cfg.ENCODER_PATH, cfg.SCALER_PATH, cfg.MODEL_PATH, cfg.METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"model artifact '{path.name}' not found in {cfg.MODEL_V2_DIR}. "
                "Run training/scripts/03_train_xgboost.py to build the v2 artifacts."
            )
    encoder = joblib.load(cfg.ENCODER_PATH)
    scaler = joblib.load(cfg.SCALER_PATH)
    model = joblib.load(cfg.MODEL_PATH)
    metadata = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))

    if metadata["feature_order"] != spec.FEATURE_ORDER:
        raise RuntimeError(
            "feature order in model_metadata.json does not match "
            "fraudlens.core.features.FEATURE_ORDER; the model and the app are "
            "out of sync. Retrain before serving predictions."
        )
    return encoder, scaler, model, metadata


def build_model_row(tx: TransactionInput) -> tuple[dict, float, float, float]:
    """Turn UI input into the model's feature dict.

    Returns ``(row, distance_km, merch_lat, merch_long)``. The row's keys are in
    :data:`fraudlens.core.features.FEATURE_ORDER` order.
    """
    # Merchant coordinates cannot be looked up from a merchant identifier (see
    # fraudlens/core/geo.py for the measurement). They are drawn from the
    # measured offset distribution around the customer, seeded from this
    # transaction's own values so the result is stable for identical input.
    seed_parts = {
        "gender": tx.gender, "age": tx.age, "job": tx.job,
        "state": tx.state, "city": tx.city, "street": tx.street, "zip": tx.zip_code,
        "amount": round(float(tx.amount), 2), "category": tx.category,
        "day_of_week": tx.day_of_week, "month": tx.month_name, "date": tx.date,
        "hour": tx.hour, "minute": tx.minute, "second": tx.second,
    }
    distance_km, merch_lat, merch_long = resolve_transaction_distance(
        tx.lat, tx.long, seed_parts
    )

    row = {
        "category": tx.category,
        "amt": float(tx.amount),
        "gender": tx.gender,
        "state": tx.state,
        "city": tx.city,
        "street": tx.street,
        "zip": int(tx.zip_code),
        "job": tx.job,
        "age": int(tx.age),
        "day_of_week": tx.day_of_week,
        "transaction_hour": int(tx.hour),
        "transaction_min": int(tx.minute),
        "transaction_seconds": int(tx.second),
        "transaction_date": int(tx.date),
        "transaction_month": spec.MONTH_NAME_TO_NUMBER[tx.month_name],
        "transaction_distance": float(distance_km),
    }
    # Build in the declared order rather than relying on dict literal order.
    row = {name: row[name] for name in spec.FEATURE_ORDER}
    return row, distance_km, merch_lat, merch_long


def predict(tx: TransactionInput) -> Prediction:
    """Run the full inference chain for one transaction."""
    encoder, scaler, model, _ = load_artifacts()
    row, distance_km, merch_lat, merch_long = build_model_row(tx)

    frame = pd.DataFrame([row], columns=spec.FEATURE_ORDER)
    frame[spec.CATEGORICAL_FEATURES] = encoder.transform(
        frame[spec.CATEGORICAL_FEATURES]
    )
    scaled = scaler.transform(frame[spec.FEATURE_ORDER])

    label = int(model.predict(scaled)[0])
    proba = float(model.predict_proba(scaled)[0][1])

    return Prediction(
        is_fraud=bool(label == 1),
        fraud_probability=proba,
        transaction_distance_km=distance_km,
        merchant_lat=merch_lat,
        merchant_long=merch_long,
        model_row=row,
    )
