"""Dataset-derived reference data and the exactness of the location cascade."""

from __future__ import annotations

import pytest

from fraudlens.core import features as spec
from fraudlens.core import reference as ref


def test_states_are_non_empty_and_sorted():
    states = ref.states()
    assert states and states == sorted(states)


def test_cascade_narrows_at_every_level():
    state = ref.states()[0]
    cities = ref.cities(state)
    assert cities
    street = ref.streets(state, cities[0])
    assert street
    assert ref.zips(state, cities[0], street[0])


@pytest.mark.parametrize("args", [
    ("", "", ""), ("NC", "", ""), ("NC", "Clarkton", ""),
])
def test_cascade_returns_nothing_without_its_parents(args):
    assert ref.zips(*args) == []


def test_every_recorded_address_resolves_to_exactly_one_zip_and_coordinate():
    # The audit measured this as true for 100% of the recorded locations; if a
    # future reference build breaks it, the UI contract changes and this fails.
    df = ref.locations()
    grouped = df.groupby(["state", "city", "street"], observed=True)
    assert grouped["zip"].nunique().max() == 1
    assert grouped["lat"].nunique().max() == 1
    assert grouped["long"].nunique().max() == 1


def test_coordinates_resolve_for_a_real_address():
    state = ref.states()[0]
    city = ref.cities(state)[0]
    street = ref.streets(state, city)[0]
    zip_code = ref.zips(state, city, street)[0]
    coords = ref.resolve_coordinates(state, city, street, zip_code)
    assert coords is not None
    lat, long = coords
    assert -90 <= lat <= 90 and -180 <= long <= 180


def test_coordinates_are_none_for_an_invented_address():
    assert ref.resolve_coordinates("NC", "Clarkton", "No Such Street", "28433") is None
    assert ref.resolve_coordinates("", "", "", "") is None


def test_zip_codes_keep_their_leading_zero():
    zips = ref.locations()["zip"]
    assert zips.str.len().eq(5).all()


def test_option_lists_come_from_the_training_vocabulary():
    vocab = ref.vocabulary()
    assert ref.jobs() == list(vocab["jobs"])
    assert sorted(ref.categories()) == sorted(vocab["categories"])
    assert sorted(ref.genders()) == sorted(vocab["genders"])


def test_categories_are_display_ordered_and_labelled():
    labels = [ref.category_label(v) for v in ref.categories()]
    assert labels == sorted(labels)
    assert ref.gender_label("F") == "Female"


def test_measured_merchant_offset_is_recorded():
    offset = ref.merchant_offset()
    assert offset  # written by the processed build from the raw archive


def test_no_option_list_is_hardcoded_in_the_module():
    source = (ref.__file__ and open(ref.__file__, encoding="utf-8").read()) or ""
    for banned in ("Alabama", "'AK'", "grocery_pos"):
        assert banned not in source


def test_vocabulary_covers_every_categorical_feature_the_form_offers():
    vocab = ref.vocabulary()
    for feature in ("jobs", "categories", "genders"):
        assert vocab[feature]
    assert set(spec.CATEGORICAL_FEATURES) == {
        "category", "gender", "state", "city", "street", "job", "day_of_week"
    }
