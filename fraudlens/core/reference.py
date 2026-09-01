"""Dataset-derived reference lookups for the form.

Every option list the UI offers comes from ``Data/processed/reference/``, which
is generated from the raw archive by ``training/scripts/02_build_processed.py``.
Nothing here is hardcoded and no external location source is used.

The audit measured State + City + Street as resolving a single ZIP and a single
customer coordinate pair for 100% of the 983 distinct locations, so the cascade
below is an exact lookup.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config as cfg
from . import features as spec


@lru_cache(maxsize=1)
def locations() -> pd.DataFrame:
    if not cfg.LOCATIONS_CSV.exists():
        raise FileNotFoundError(
            f"{cfg.LOCATIONS_CSV} is missing. Run "
            "training/scripts/02_build_processed.py to generate reference data."
        )
    df = pd.read_csv(cfg.LOCATIONS_CSV, dtype={"zip": str})
    df["zip"] = df["zip"].str.zfill(5)
    return df


@lru_cache(maxsize=1)
def vocabulary() -> dict:
    if not cfg.VOCAB_JSON.exists():
        raise FileNotFoundError(
            f"{cfg.VOCAB_JSON} is missing. Run "
            "training/scripts/02_build_processed.py to generate reference data."
        )
    return json.loads(cfg.VOCAB_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def merchant_offset() -> dict:
    if not cfg.GEO_OFFSET_JSON.exists():
        raise FileNotFoundError(f"{cfg.GEO_OFFSET_JSON} is missing.")
    return json.loads(cfg.GEO_OFFSET_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Location cascade
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def states() -> list[str]:
    return sorted(locations()["state"].unique().tolist())


@lru_cache(maxsize=64)
def cities(state: str) -> list[str]:
    if not state:
        return []
    df = locations()
    return sorted(df.loc[df["state"] == state, "city"].unique().tolist())


@lru_cache(maxsize=512)
def streets(state: str, city: str) -> list[str]:
    if not state or not city:
        return []
    df = locations()
    mask = (df["state"] == state) & (df["city"] == city)
    return sorted(df.loc[mask, "street"].unique().tolist())


@lru_cache(maxsize=2048)
def zips(state: str, city: str, street: str) -> list[str]:
    if not state or not city or not street:
        return []
    df = locations()
    mask = (df["state"] == state) & (df["city"] == city) & (df["street"] == street)
    return sorted(df.loc[mask, "zip"].unique().tolist())


@lru_cache(maxsize=2048)
def resolve_coordinates(
    state: str, city: str, street: str, zip_code: str
) -> tuple[float, float] | None:
    """Customer ``(lat, long)`` for a fully specified location, else ``None``."""
    if not all((state, city, street, zip_code)):
        return None
    df = locations()
    mask = (
        (df["state"] == state)
        & (df["city"] == city)
        & (df["street"] == street)
        & (df["zip"] == str(zip_code).zfill(5))
    )
    match = df.loc[mask]
    if len(match) != 1:
        # The cascade is exact in the reference table, so this only fires if a
        # stale downstream selection survived a parent change.
        return None
    row = match.iloc[0]
    return float(row["lat"]), float(row["long"])


# ---------------------------------------------------------------------------
# Other option lists
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def jobs() -> list[str]:
    return list(vocabulary()["jobs"])


@lru_cache(maxsize=1)
def categories() -> list[str]:
    """Model values for ``category``, ordered by their display label."""
    values = vocabulary()["categories"]
    return sorted(values, key=lambda v: spec.CATEGORY_DISPLAY.get(v, v))


@lru_cache(maxsize=1)
def genders() -> list[str]:
    return list(vocabulary()["genders"])


def category_label(value: str) -> str:
    return spec.CATEGORY_DISPLAY.get(value, value)


def gender_label(value: str) -> str:
    return spec.GENDER_DISPLAY.get(value, value)
