"""Read-only audit of the raw FraudLens dataset.

Reads ``Data/archive.zip`` without extracting or modifying it and measures the
data facts the rebuild depends on:

* schema, row counts, nulls, duplicates, class balance
* whether ``merchant`` (or ``merchant`` + ``category``) determines a single
  merchant coordinate pair
* whether the location cascade State -> City -> Street -> ZIP is unambiguous
  and whether it resolves a single customer coordinate pair

The findings are written to ``training/reports/raw_audit.json`` so later steps
can cite measured numbers instead of assumptions. Nothing in the raw archive is
written to.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Data" / "archive.zip"
REPORT_DIR = ROOT / "training" / "reports"

# Columns the audit needs. Reading a subset keeps peak memory well below the
# cost of the full 23-column frame for a ~1.3M row file.
AUDIT_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "gender", "street", "city", "state", "zip", "lat", "long",
    "city_pop", "job", "dob", "merch_lat", "merch_long", "is_fraud",
]


def read_raw(member: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load one CSV member of the raw archive read-only."""
    with zipfile.ZipFile(ARCHIVE) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, usecols=usecols, low_memory=False)


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorised form of the approved Haversine method (kilometres)."""
    radius_km = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def group_cardinality(df: pd.DataFrame, keys: list[str], value: list[str]) -> dict:
    """How many distinct `value` tuples exist per `keys` tuple."""
    counts = df.groupby(keys, observed=True)[value].nunique()
    # For a multi-column value, take the row-wise max distinct count.
    per_group = counts.max(axis=1) if counts.ndim > 1 else counts
    return {
        "groups": int(per_group.shape[0]),
        "groups_with_one": int((per_group == 1).sum()),
        "groups_with_many": int((per_group > 1).sum()),
        "max_distinct": int(per_group.max()),
        "mean_distinct": round(float(per_group.mean()), 4),
        "pct_unambiguous": round(float((per_group == 1).mean() * 100), 4),
    }


def audit_member(member: str) -> dict:
    df = read_raw(member, AUDIT_COLS)
    out: dict = {"member": member, "rows": int(len(df))}

    out["columns"] = list(df.columns)
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    out["nulls"] = {c: int(n) for c, n in df.isnull().sum().items() if n}
    out["duplicate_rows_full"] = int(df.duplicated().sum())

    fraud = int(df["is_fraud"].sum())
    out["class_balance"] = {
        "legit": int(len(df) - fraud),
        "fraud": fraud,
        "fraud_pct": round(fraud / len(df) * 100, 4),
    }

    out["unique_counts"] = {
        c: int(df[c].nunique())
        for c in ["cc_num", "merchant", "category", "job", "state", "city",
                  "street", "zip", "gender"]
    }

    ts = pd.to_datetime(df["trans_date_trans_time"])
    out["time_span"] = {"min": str(ts.min()), "max": str(ts.max())}
    out["seconds_present"] = {
        "distinct_second_values": int(ts.dt.second.nunique()),
        "nonzero_second_pct": round(float((ts.dt.second != 0).mean() * 100), 4),
    }

    # --- Open item: does merchant identity fix a merchant coordinate? -------
    out["merchant_coordinate"] = {
        "by_merchant": group_cardinality(df, ["merchant"], ["merch_lat", "merch_long"]),
        "by_merchant_category": group_cardinality(
            df, ["merchant", "category"], ["merch_lat", "merch_long"]
        ),
    }

    # If merchant coordinates are actually generated near the customer, the
    # offset between the two coordinate pairs will be small and bounded.
    d_lat = (df["merch_lat"] - df["lat"]).abs()
    d_lon = (df["merch_long"] - df["long"]).abs()
    dist = haversine_vec(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    out["customer_vs_merchant_offset"] = {
        "abs_dlat": {"min": round(float(d_lat.min()), 6), "max": round(float(d_lat.max()), 6),
                     "mean": round(float(d_lat.mean()), 6)},
        "abs_dlon": {"min": round(float(d_lon.min()), 6), "max": round(float(d_lon.max()), 6),
                     "mean": round(float(d_lon.mean()), 6)},
        "haversine_km": {
            "min": round(float(dist.min()), 4),
            "p05": round(float(dist.quantile(0.05)), 4),
            "median": round(float(dist.median()), 4),
            "p95": round(float(dist.quantile(0.95)), 4),
            "max": round(float(dist.max()), 4),
            "mean": round(float(dist.mean()), 4),
            "std": round(float(dist.std()), 4),
        },
        "by_fraud_class_median_km": {
            "legit": round(float(dist[df["is_fraud"] == 0].median()), 4),
            "fraud": round(float(dist[df["is_fraud"] == 1].median()), 4),
        },
    }

    # --- Open item: is the location cascade unambiguous? -------------------
    out["location_cascade"] = {
        "state_to_city": group_cardinality(df, ["state"], ["city"]),
        "state_city_to_street": group_cardinality(df, ["state", "city"], ["street"]),
        "state_city_street_to_zip": group_cardinality(
            df, ["state", "city", "street"], ["zip"]
        ),
        "state_city_street_zip_to_coords": group_cardinality(
            df, ["state", "city", "street", "zip"], ["lat", "long"]
        ),
        "street_to_ccnum": group_cardinality(df, ["street"], ["cc_num"]),
    }

    # Customer identity: is cc_num really stronger than first+last?
    out["identity"] = {
        "ccnum_to_street": group_cardinality(df, ["cc_num"], ["street"]),
        "ccnum_to_coords": group_cardinality(df, ["cc_num"], ["lat", "long"]),
        "ccnum_to_job": group_cardinality(df, ["cc_num"], ["job"]),
    }

    # Numeric ranges that drive UI bounds.
    age = ts.dt.year - pd.to_datetime(df["dob"]).dt.year
    out["numeric_ranges"] = {
        "amt": {"min": round(float(df["amt"].min()), 2), "max": round(float(df["amt"].max()), 2),
                "median": round(float(df["amt"].median()), 2)},
        "age": {"min": int(age.min()), "max": int(age.max()), "median": int(age.median())},
        "zip": {"min": int(df["zip"].min()), "max": int(df["zip"].max())},
        "city_pop": {"min": int(df["city_pop"].min()), "max": int(df["city_pop"].max())},
    }

    out["categories"] = sorted(df["category"].unique().tolist())
    out["genders"] = sorted(df["gender"].unique().tolist())
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_bytes": ARCHIVE.stat().st_size,
        "members": {},
    }
    for member in ("fraudTrain.csv", "fraudTest.csv"):
        print(f"auditing {member} ...", flush=True)
        report["members"][member] = audit_member(member)

    out_path = REPORT_DIR / "raw_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
