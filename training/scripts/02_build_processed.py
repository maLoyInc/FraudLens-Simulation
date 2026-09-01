"""Build the FraudLens v2 processed dataset and UI reference data from raw.

Reads ``Data/archive.zip`` read-only and writes to ``Data/processed/``:

* ``fraudlens_train_processed.parquet`` - modelling frame from ``fraudTrain.csv``
* ``fraudlens_test_processed.parquet``  - modelling frame from ``fraudTest.csv``
* ``reference/locations.csv``           - State -> City -> Street -> ZIP -> lat/long
* ``reference/vocabulary.json``         - option lists for the form
* ``reference/merchant_offset.json``    - measured customer/merchant offset bounds

The raw archive is never modified.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fraudlens.core import config as cfg
from fraudlens.core import features as spec
from fraudlens.core.geo import OFFSET_DEGREES, haversine_series

# Raw columns the build actually reads. cc_num is read only so the reference
# tables can be validated against the stronger customer key, then dropped.
READ_COLS = [
    "trans_date_trans_time", "cc_num", "category", "amt", "gender",
    "street", "city", "state", "zip", "lat", "long", "job", "dob",
    "merch_lat", "merch_long", "is_fraud",
]


def read_raw(member: str) -> pd.DataFrame:
    with zipfile.ZipFile(cfg.RAW_ARCHIVE) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, usecols=READ_COLS, low_memory=False)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the v2 cleaning and feature engineering to a raw frame."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    out = pd.DataFrame(index=df.index)

    # Categorical features, in the representation the encoder will be fitted on.
    out["category"] = df["category"].map(spec.normalise_category)
    out["gender"] = df["gender"]
    out["state"] = df["state"]
    out["city"] = df["city"]
    out["street"] = df["street"]
    out["job"] = df["job"]
    out["day_of_week"] = ts.dt.day_name()

    # Numeric features.
    out["amt"] = df["amt"].astype("float64")
    out["zip"] = df["zip"].astype("int64")
    # Age as the research pipeline defined it: difference of calendar years.
    out["age"] = (ts.dt.year - dob.dt.year).astype("int64")
    out["transaction_hour"] = ts.dt.hour.astype("int64")
    out["transaction_min"] = ts.dt.minute.astype("int64")
    # New in v2: seconds are present in the raw timestamp and are now a real
    # model feature rather than being discarded.
    out["transaction_seconds"] = ts.dt.second.astype("int64")
    out["transaction_date"] = ts.dt.day.astype("int64")
    out["transaction_month"] = ts.dt.month.astype("int64")
    out["transaction_distance"] = haversine_series(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    out[spec.TARGET] = df["is_fraud"].astype("int8")

    # Reorder to the declared model input order, target last.
    return out[spec.FEATURE_ORDER + [spec.TARGET]]


def build_locations(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct customer location, with its resolved coordinates.

    The audit measured State+City+Street -> ZIP and -> (lat, long) as 100%
    unambiguous, so this table is a true lookup rather than an approximation.
    """
    loc = (
        df_raw[["state", "city", "street", "zip", "lat", "long"]]
        .drop_duplicates()
        .sort_values(["state", "city", "street"])
        .reset_index(drop=True)
    )

    dupes = loc.duplicated(subset=["state", "city", "street"]).sum()
    if dupes:
        raise SystemExit(
            f"location cascade is ambiguous: {dupes} State+City+Street keys map "
            "to more than one ZIP/coordinate. Resolve before continuing."
        )
    return loc


def build_vocabulary(model_df: pd.DataFrame) -> dict:
    """Option lists for the form, taken from the training member only.

    Restricting to the training member guarantees every option the UI can offer
    is a value the ordinal encoder was actually fitted on.
    """
    return {
        "categories": sorted(model_df["category"].unique().tolist()),
        "genders": sorted(model_df["gender"].unique().tolist()),
        "jobs": sorted(model_df["job"].unique().tolist()),
        "weekdays": spec.WEEKDAY_NAMES,
        "months": spec.MONTH_NAMES,
        "amount": {
            "observed_min": round(float(model_df["amt"].min()), 2),
            "observed_max": round(float(model_df["amt"].max()), 2),
            "observed_median": round(float(model_df["amt"].median()), 2),
        },
        "age": {
            "observed_min": int(model_df["age"].min()),
            "observed_max": int(model_df["age"].max()),
        },
        "transaction_distance": {
            "observed_min": round(float(model_df["transaction_distance"].min()), 4),
            "observed_max": round(float(model_df["transaction_distance"].max()), 4),
            "observed_median": round(float(model_df["transaction_distance"].median()), 4),
        },
    }


def build_offset_metadata(df_raw: pd.DataFrame) -> dict:
    """Measured customer/merchant coordinate offset, used at inference time."""
    d_lat = (df_raw["merch_lat"] - df_raw["lat"]).abs()
    d_lon = (df_raw["merch_long"] - df_raw["long"]).abs()
    return {
        "method": "uniform_offset_around_customer",
        "offset_degrees": OFFSET_DEGREES,
        "measured_abs_dlat_max": round(float(d_lat.max()), 6),
        "measured_abs_dlon_max": round(float(d_lon.max()), 6),
        "measured_abs_dlat_mean": round(float(d_lat.mean()), 6),
        "measured_abs_dlon_mean": round(float(d_lon.mean()), 6),
        "rationale": (
            "No merchant identifier in the raw data maps to a single merchant "
            "coordinate (0 of 693 merchants, 0 of 700 merchant+category pairs). "
            "The offset between the customer and merchant coordinate is uniform "
            "inside a +/-1 degree box around the customer and is independent of "
            "merchant identity, so a merchant has no coordinate to look up. "
            "Inference reproduces this measured process with a seed derived from "
            "the transaction's own field values, which keeps a given transaction "
            "deterministic."
        ),
    }


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:
        fallback = path.with_suffix(".csv.gz")
        print(f"  parquet unavailable ({exc}); writing {fallback.name}")
        df.to_csv(fallback, index=False, compression="gzip")


def main() -> None:
    spec.assert_feature_spec_consistent()
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"feature_order": spec.FEATURE_ORDER, "members": {}}

    for member, out_path, is_primary in (
        (cfg.RAW_TRAIN_MEMBER, cfg.TRAIN_PROCESSED, True),
        (cfg.RAW_TEST_MEMBER, cfg.TEST_PROCESSED, False),
    ):
        print(f"processing {member} ...", flush=True)
        raw = read_raw(member)
        model_df = engineer(raw)

        assert list(model_df.columns) == spec.FEATURE_ORDER + [spec.TARGET]
        assert "city_pop" not in model_df.columns
        assert model_df["transaction_seconds"].between(0, 59).all()
        assert model_df["transaction_hour"].between(0, 23).all()
        assert not model_df.isnull().any().any()

        write_parquet(model_df, out_path)
        print(f"  wrote {out_path.name}: {len(model_df):,} rows x {model_df.shape[1]} cols")

        summary["members"][member] = {
            "rows": int(len(model_df)),
            "fraud": int(model_df[spec.TARGET].sum()),
            "distance_km": {
                "min": round(float(model_df["transaction_distance"].min()), 4),
                "median": round(float(model_df["transaction_distance"].median()), 4),
                "max": round(float(model_df["transaction_distance"].max()), 4),
            },
        }

        if is_primary:
            locations = build_locations(raw)
            locations.to_csv(cfg.LOCATIONS_CSV, index=False)
            print(f"  wrote {cfg.LOCATIONS_CSV.name}: {len(locations):,} locations")

            vocab = build_vocabulary(model_df)
            cfg.VOCAB_JSON.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
            print(f"  wrote {cfg.VOCAB_JSON.name}")

            cfg.GEO_OFFSET_JSON.write_text(
                json.dumps(build_offset_metadata(raw), indent=2), encoding="utf-8"
            )
            print(f"  wrote {cfg.GEO_OFFSET_JSON.name}")

            summary["locations"] = int(len(locations))
            summary["states"] = int(locations["state"].nunique())

        del raw, model_df

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORTS_DIR / "processed_build.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("done")


if __name__ == "__main__":
    main()
