"""Haversine distance and merchant-coordinate resolution."""

from __future__ import annotations

import math

from fraudlens.core import geo


def test_zero_distance_for_identical_points():
    assert geo.haversine(40.7128, -74.0060, 40.7128, -74.0060) == 0.0


def test_known_reference_distance_nyc_to_london():
    # Great-circle NYC (JFK) to London (LHR) is ~5555 km at R = 6371 km.
    km = geo.haversine(40.6413, -73.7781, 51.4700, -0.4543)
    assert math.isclose(km, 5555, rel_tol=0.01)


def test_known_reference_distance_one_degree_of_latitude():
    # One degree of latitude is pi * R / 180 km regardless of longitude.
    expected = math.pi * geo.EARTH_RADIUS_KM / 180
    km = geo.haversine(0.0, 10.0, 1.0, 10.0)
    assert math.isclose(km, expected, rel_tol=1e-9)


def test_distance_is_symmetric():
    a = geo.haversine(34.05, -118.24, 41.88, -87.63)
    b = geo.haversine(41.88, -87.63, 34.05, -118.24)
    assert math.isclose(a, b, rel_tol=1e-12)


def test_earth_radius_is_the_approved_value():
    assert geo.EARTH_RADIUS_KM == 6371


SEED_PARTS = {"street": "9333 Ross Drive", "amount": 120.5, "hour": 22}


def test_merchant_coords_are_deterministic():
    first = geo.resolve_merchant_coords(34.4762, -78.6534, SEED_PARTS)
    second = geo.resolve_merchant_coords(34.4762, -78.6534, SEED_PARTS)
    assert first == second


def test_merchant_coords_change_with_the_seed():
    a = geo.resolve_merchant_coords(34.4762, -78.6534, SEED_PARTS)
    b = geo.resolve_merchant_coords(34.4762, -78.6534, {**SEED_PARTS, "hour": 3})
    assert a != b


def test_merchant_offset_stays_inside_the_measured_box():
    # The audit measured the offset as uniform within +/- 1 degree on each axis.
    for index in range(400):
        lat, long = 34.4762, -78.6534
        m_lat, m_long = geo.resolve_merchant_coords(lat, long, {"i": index})
        assert abs(m_lat - lat) <= geo.OFFSET_DEGREES
        assert abs(m_long - long) <= geo.OFFSET_DEGREES


def test_resolve_transaction_distance_matches_haversine():
    km, m_lat, m_long = geo.resolve_transaction_distance(34.4762, -78.6534, SEED_PARTS)
    assert math.isclose(km, geo.haversine(34.4762, -78.6534, m_lat, m_long),
                        rel_tol=1e-12)
    assert 0.0 < km < 200.0
