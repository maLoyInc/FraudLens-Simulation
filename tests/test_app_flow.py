"""End-to-end UI flow through the real Streamlit runtime.

Uses ``streamlit.testing.v1.AppTest``, which executes ``fraudlens/app.py`` in a
genuine script-run context, so step transitions, validation feedback and the
cascading resets are exercised rather than described.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from fraudlens.core import reference as ref

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1]
          / "fraudlens" / "app.py")
TIMEOUT = 120


def start() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    return app.run()


def address() -> tuple[str, str, str, str]:
    state = ref.states()[0]
    city = ref.cities(state)[0]
    street = ref.streets(state, city)[0]
    zip_code = ref.zips(state, city, street)[0]
    return state, city, street, zip_code


def markup(app: AppTest) -> str:
    return "".join(block.value for block in app.markdown)


def value(app: AppTest, key: str):
    """Session value, treating an absent key as empty.

    A reset drops the key; the widget then re-registers it as ``None`` on the
    next run because it renders with ``index=None``. Both mean "cleared".
    """
    try:
        return app.session_state[key]
    except KeyError:
        return None


@pytest.fixture(scope="module")
def customer_done() -> AppTest:
    app = start()
    app.selectbox("fl_gender").set_value(ref.genders()[0])
    app.number_input("fl_age").set_value(42)
    app.selectbox("fl_job").set_value(ref.jobs()[0])
    app.button("fl_next_1").click().run()
    return app


def test_app_starts_on_step_one_without_exceptions():
    app = start()
    assert not app.exception
    assert app.session_state["fl_step"] == 1
    assert "Customer" in markup(app)


def test_incomplete_step_one_blocks_and_reports_every_field():
    app = start()
    app.button("fl_next_1").click().run()
    assert app.session_state["fl_step"] == 1
    body = markup(app)
    assert "Gender is required." in body
    assert "Age is required." in body
    assert "Job is required." in body


def test_complete_step_one_advances(customer_done):
    assert not customer_done.exception
    assert customer_done.session_state["fl_step"] == 2


def test_card_number_is_never_requested():
    app = start()
    body = markup(app).lower()
    assert "card number" in body      # explained as deliberately absent
    assert not [w for w in app.text_input if "card" in w.label.lower()]
    assert "cc_num" not in body


def test_location_cascade_resets_downstream_selections():
    state, city, street, zip_code = address()
    app = start()
    app.session_state["fl_step"] = 2
    app.run()
    app.selectbox("fl_state").set_value(state).run()
    app.selectbox("fl_city").set_value(city).run()
    app.selectbox("fl_street").set_value(street).run()
    assert app.session_state["fl_zip"] == zip_code

    other = [s for s in ref.states() if s != state][0]
    app.selectbox("fl_state").set_value(other).run()
    assert value(app, "fl_city") is None
    assert value(app, "fl_street") is None
    assert value(app, "fl_zip") is None


def test_changing_city_resets_only_street_and_zip():
    state = ref.states()[0]
    cities = [c for c in ref.cities(state) if ref.streets(state, c)]
    if len(cities) < 2:
        state = next(s for s in ref.states() if len(ref.cities(s)) > 1)
        cities = [c for c in ref.cities(state) if ref.streets(state, c)]
    app = start()
    app.session_state["fl_step"] = 2
    app.run()
    app.selectbox("fl_state").set_value(state).run()
    app.selectbox("fl_city").set_value(cities[0]).run()
    app.selectbox("fl_street").set_value(ref.streets(state, cities[0])[0]).run()
    assert value(app, "fl_zip") is not None

    app.selectbox("fl_city").set_value(cities[1]).run()
    assert app.session_state["fl_state"] == state
    assert app.session_state["fl_city"] == cities[1]
    assert value(app, "fl_street") is None
    assert value(app, "fl_zip") is None


def test_zip_and_coordinates_are_derived_not_typed():
    state, city, street, zip_code = address()
    app = start()
    app.session_state["fl_step"] = 2
    app.run()
    app.selectbox("fl_state").set_value(state).run()
    app.selectbox("fl_city").set_value(city).run()
    app.selectbox("fl_street").set_value(street).run()
    body = markup(app)
    assert zip_code in body
    lat, long = ref.resolve_coordinates(state, city, street, zip_code)
    assert f"{lat:.4f}, {long:.4f}" in body
    # No free-text or numeric entry for coordinates anywhere on the step.
    labels = [w.label.lower() for w in app.number_input] + \
             [w.label.lower() for w in app.text_input]
    assert not any("lat" in l or "long" in l for l in labels)


def test_amount_field_normalises_to_the_display_form():
    app = start()
    app.session_state["fl_step"] = 3
    app.run()
    app.text_input("fl_amount_raw").set_value("34234").run()
    assert app.session_state["fl_amount_raw"] == "34,234.00"


def test_transaction_step_reports_invalid_amount():
    app = start()
    app.session_state["fl_step"] = 3
    app.run()
    app.text_input("fl_amount_raw").set_value("abc").run()
    app.button("fl_next_3").click().run()
    assert app.session_state["fl_step"] == 3
    assert "Amount may only contain digits" in markup(app)


def fill_everything(app: AppTest) -> AppTest:
    state, city, street, zip_code = address()
    app.selectbox("fl_gender").set_value(ref.genders()[0])
    app.number_input("fl_age").set_value(42)
    app.selectbox("fl_job").set_value(ref.jobs()[0])
    app.button("fl_next_1").click().run()

    app.selectbox("fl_state").set_value(state).run()
    app.selectbox("fl_city").set_value(city).run()
    app.selectbox("fl_street").set_value(street).run()
    app.button("fl_next_2").click().run()

    app.text_input("fl_amount_raw").set_value("34,234.00")
    app.selectbox("fl_category").set_value(ref.categories()[0])
    app.selectbox("fl_weekday").set_value("Monday")
    app.selectbox("fl_month").set_value("March")
    app.selectbox("fl_date").set_value(14)
    app.selectbox("fl_hour").set_value(22)
    app.selectbox("fl_minute").set_value(5)
    app.selectbox("fl_second").set_value(9)
    app.button("fl_next_3").click().run()
    return app


def test_full_flow_reviews_and_predicts():
    state, city, street, zip_code = address()
    app = fill_everything(start())
    assert not app.exception
    assert app.session_state["fl_step"] == 4

    review = markup(app)
    for expected in ("Review", "34,234.00", street, zip_code, "22:05:09",
                     "March", "Monday", " km"):
        assert expected in review

    app.button("fl_predict").click().run()
    assert not app.exception
    assert app.session_state["fl_step"] == 5

    prediction = app.session_state["fl_prediction"]
    assert 0.0 <= prediction.fraud_probability <= 1.0
    result = markup(app)
    assert "Model classification" in result
    assert "Fraud probability" in result
    assert "Educational use only" in result


def test_result_step_offers_a_clean_reset():
    app = fill_everything(start())
    app.button("fl_predict").click().run()
    app.button("fl_reset").click().run()
    assert app.session_state["fl_step"] == 1
    assert app.session_state["fl_prediction"] is None
    assert value(app, "fl_amount_raw") in (None, "")
    assert value(app, "fl_state") is None


def test_review_refuses_to_score_an_incomplete_record():
    app = start()
    app.session_state["fl_step"] = 4
    app.run()
    assert not app.exception
    assert "cannot be reviewed yet" in markup(app)
    assert not [b for b in app.button if b.key == "fl_predict"]


def test_result_step_without_a_prediction_is_handled():
    app = start()
    app.session_state["fl_step"] = 5
    app.run()
    assert not app.exception
    assert "Nothing to show" in markup(app)


def test_stepper_marks_progress():
    app = fill_everything(start())
    body = markup(app)
    assert "fl-step--active" in body
    assert "fl-step--done" in body


def test_no_default_streamlit_alert_blocks_are_used():
    app = start()
    app.button("fl_next_1").click().run()
    assert not app.error
    assert not app.warning
    assert not app.exception

