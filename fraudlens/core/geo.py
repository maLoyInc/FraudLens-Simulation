"""Geographic helpers.

Holds the approved Haversine implementation and the empirically-derived rule
for resolving merchant coordinates at inference time.
"""

from __future__ import annotations

import hashlib
import json
from math import atan2, cos, radians, sin, sqrt

import numpy as np

EARTH_RADIUS_KM = 6371


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres.

    Mathematically identical to the formula used to build the research feature
    ``transaction_distance``; kept as a scalar function so it can be unit
    tested against known reference distances.
    """
    R = EARTH_RADIUS_KM
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def haversine_series(lat1, lon1, lat2, lon2):
    """Vectorised Haversine for building the training feature column.

    Returns the same values as :func:`haversine` applied row by row, but fast
    enough for the full 1.3M-row raw file.
    """
    R = EARTH_RADIUS_KM
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype="float64"))
                              for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Merchant coordinate resolution
# ---------------------------------------------------------------------------
# Measured on the raw archive (see training/reports/raw_audit.json):
#
#   merchant                -> merchant coordinate : 0 of 693 unambiguous
#   merchant + category     -> merchant coordinate : 0 of 700 unambiguous
#   |merch_lat  - lat |     : min 0.000000  max 0.999999  mean 0.500263
#   |merch_long - long|     : min 0.000000  max 0.999997  mean 0.500337
#
# The offset between the customer and merchant coordinate is uniform inside a
# +/-1 degree box around the customer, independent of merchant identity. A
# merchant therefore has no stable coordinate that could be looked up.
#
# At inference we reproduce that measured process instead of inventing a fixed
# coordinate: the offset is drawn from the same uniform box, seeded from the
# transaction's own field values so a given transaction always resolves to the
# same merchant coordinate and the same distance.
OFFSET_DEGREES = 1.0


def _seed_from(parts: dict) -> int:
    """Stable 64-bit seed from the transaction's field values.

    Uses a content hash rather than Python's ``hash()`` so the seed is
    identical across processes and runs (``hash()`` is salted per process).
    """
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def resolve_merchant_coords(
    lat: float,
    long: float,
    seed_parts: dict,
    offset_degrees: float = OFFSET_DEGREES,
) -> tuple[float, float]:
    """Resolve a merchant coordinate for a new transaction.

    Draws a uniform offset inside the measured +/-``offset_degrees`` box around
    the customer coordinate. Deterministic for a given ``seed_parts``.
    """
    rng = np.random.default_rng(_seed_from(seed_parts))
    d_lat, d_long = rng.uniform(-offset_degrees, offset_degrees, size=2)
    return float(lat + d_lat), float(long + d_long)


def resolve_transaction_distance(
    lat: float, long: float, seed_parts: dict,
    offset_degrees: float = OFFSET_DEGREES,
) -> tuple[float, float, float]:
    """Return ``(distance_km, merch_lat, merch_long)`` for a new transaction."""
    merch_lat, merch_long = resolve_merchant_coords(lat, long, seed_parts, offset_degrees)
    return haversine(lat, long, merch_lat, merch_long), merch_lat, merch_long
