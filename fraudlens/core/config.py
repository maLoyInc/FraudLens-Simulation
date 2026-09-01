"""Paths and shared constants for FraudLens.

Single place that knows where things live, so training scripts, the app and the
tests all agree.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Immutable research input. Never written to.
RAW_ARCHIVE = ROOT / "Data" / "archive.zip"
RAW_TRAIN_MEMBER = "fraudTrain.csv"
RAW_TEST_MEMBER = "fraudTest.csv"

# Derived data. Safe to regenerate.
PROCESSED_DIR = ROOT / "Data" / "processed"
REFERENCE_DIR = PROCESSED_DIR / "reference"
TRAIN_PROCESSED = PROCESSED_DIR / "fraudlens_train_processed.parquet"
TEST_PROCESSED = PROCESSED_DIR / "fraudlens_test_processed.parquet"

LOCATIONS_CSV = REFERENCE_DIR / "locations.csv"
VOCAB_JSON = REFERENCE_DIR / "vocabulary.json"
GEO_OFFSET_JSON = REFERENCE_DIR / "merchant_offset.json"

# Model artifacts. Legacy artifacts stay untouched in App/.
MODELS_DIR = ROOT / "models"
MODEL_V2_DIR = MODELS_DIR / "v2"
ENCODER_PATH = MODEL_V2_DIR / "ordinal_encoder.joblib"
SCALER_PATH = MODEL_V2_DIR / "standard_scaler.joblib"
MODEL_PATH = MODEL_V2_DIR / "xgboost_fraud_model.joblib"
METADATA_PATH = MODEL_V2_DIR / "model_metadata.json"

REPORTS_DIR = ROOT / "training" / "reports"

UI_STATIC_DIR = ROOT / "fraudlens" / "ui" / "static"

APP_TITLE = "FraudLens"
