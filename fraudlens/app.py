"""FraudLens - Streamlit entry point.

Run with::

    streamlit run fraudlens/app.py

Streamlit is used only as the Python runtime and widget transport; the visual
layer is the custom dark design system in ``fraudlens/ui/``. The app is
inference-only: it loads the artifacts in ``models/v2/`` and never fits or
writes a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run fraudlens/app.py` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from fraudlens.core import features as spec
from fraudlens.core import reference as ref
from fraudlens.core import validation as val
from fraudlens.inference.predictor import (
    TransactionInput,
    build_model_row,
    load_artifacts,
    predict,
)
from fraudlens.ui import components as c
from fraudlens.ui import theme

FIRST_STEP, LAST_STEP = 1, 5

# Session keys for user input, grouped by the step that owns them.
CUSTOMER_KEYS = ("fl_gender", "fl_age", "fl_job")
LOCATION_KEYS = ("fl_state", "fl_city", "fl_street", "fl_zip")
TRANSACTION_KEYS = (
    "fl_amount_raw", "fl_category", "fl_weekday", "fl_month",
    "fl_date", "fl_hour", "fl_minute", "fl_second",
)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("fl_step", FIRST_STEP)
    st.session_state.setdefault("fl_errors", {})
    st.session_state.setdefault("fl_prediction", None)
    # Streamlit discards a widget's value once that widget stops being rendered,
    # so a wizard has to keep its own copy. ``fl_data`` is that copy: widgets
    # write into it on every run and read back from it when their step is
    # revisited.
    st.session_state.setdefault("fl_data", {})


def field(key: str):
    """Current value of an input, whether or not its step is on screen."""
    if key in st.session_state:
        return st.session_state[key]
    return st.session_state["fl_data"].get(key)


def restore(*keys: str) -> None:
    """Seed widget keys from the saved copy before the widgets are created."""
    data = st.session_state["fl_data"]
    for key in keys:
        if key not in st.session_state and data.get(key) is not None:
            st.session_state[key] = data[key]


def remember(*keys: str) -> None:
    """Save the widget values of a rendered step."""
    data = st.session_state["fl_data"]
    for key in keys:
        data[key] = st.session_state.get(key)


def goto(step: int) -> None:
    st.session_state["fl_step"] = max(FIRST_STEP, min(LAST_STEP, step))


def clear_errors(step: int) -> None:
    st.session_state["fl_errors"].pop(step, None)


def set_errors(step: int, problems: list[str]) -> None:
    if problems:
        st.session_state["fl_errors"][step] = problems
    else:
        clear_errors(step)


def errors_for(step: int) -> list[str]:
    return st.session_state["fl_errors"].get(step, [])


def reset_all() -> None:
    for key in CUSTOMER_KEYS + LOCATION_KEYS + TRANSACTION_KEYS:
        st.session_state.pop(key, None)
    st.session_state["fl_data"] = {}
    st.session_state["fl_errors"] = {}
    st.session_state["fl_prediction"] = None
    goto(FIRST_STEP)


# ---------------------------------------------------------------------------
# Cascade resets
# A parent change invalidates every downstream selection, so those keys are
# dropped and their widgets fall back to the empty placeholder. Nothing stale
# is ever carried forward (PRD section 7.2, CLAUDE.md section 18).
# ---------------------------------------------------------------------------
def _drop(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)
        st.session_state["fl_data"].pop(key, None)
    st.session_state["fl_prediction"] = None



def on_state_change() -> None:
    _drop("fl_city", "fl_street", "fl_zip")


def on_city_change() -> None:
    _drop("fl_street", "fl_zip")


def on_street_change() -> None:
    _drop("fl_zip")


def on_amount_change() -> None:
    """Normalise the typed amount to the display form once it parses."""
    value, error = val.parse_amount(st.session_state.get("fl_amount_raw"))
    if value is not None and error is None:
        st.session_state["fl_amount_raw"] = val.format_amount(value)
    st.session_state["fl_prediction"] = None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def nav_row(
    step: int,
    next_label: str,
    validator,
    back_step: int | None = None,
    next_step: int | None = None,
) -> None:
    """Render Back / Continue and run the step's validator on Continue."""
    problems = errors_for(step)
    if problems:
        html(c.errors(problems))

    left, right = st.columns([1, 1], gap="small")
    if back_step is not None:
        if left.button("Back", key=f"fl_back_{step}", type="secondary",
                       width="stretch"):
            clear_errors(step)
            goto(back_step)
            st.rerun()
    if right.button(next_label, key=f"fl_next_{step}", type="primary",
                    width="stretch"):
        found = validator()
        set_errors(step, found)
        if not found:
            goto(next_step if next_step is not None else step + 1)
        st.rerun()


# ---------------------------------------------------------------------------
# Step 01 - Customer
# ---------------------------------------------------------------------------
def step_customer() -> None:
    restore(*CUSTOMER_KEYS)
    html(c.panel_head(
        "Step 01",
        "Customer",
        "Details about the cardholder. The card number is never requested: it is "
        "not a model feature and is not needed to score a transaction.",
    ))

    left, right = st.columns([1, 1], gap="medium")

    with left:
        html(c.label("Gender"))
        st.selectbox(
            "Gender", ref.genders(), key="fl_gender", index=None,
            format_func=ref.gender_label, placeholder="Select gender",
            label_visibility="collapsed",
        )

    with right:
        html(c.label("Age"))
        st.number_input(
            "Age", key="fl_age",
            min_value=spec.AGE_MIN, max_value=spec.AGE_MAX,
            value=None, step=1, placeholder="Type an age",
            label_visibility="collapsed",
        )
        html(c.hint(f"{spec.AGE_MIN}–{spec.AGE_MAX}. Values above "
                    f"{spec.AGE_MAX} are constrained to {spec.AGE_MAX}."))

    html(c.label("Job"))
    st.selectbox(
        "Job", ref.jobs(), key="fl_job", index=None,
        placeholder="Search a job title", label_visibility="collapsed",
    )
    html(c.hint(f"{len(ref.jobs())} job titles, taken from the research dataset. "
                "Start typing to filter."))

    html(c.divider())
    remember(*CUSTOMER_KEYS)
    nav_row(
        step=1,
        next_label="Continue to Location",
        validator=lambda: val.validate_customer(
            field("fl_gender"),
            val.clamp_age(field("fl_age")),
            field("fl_job"),
        ),
    )


# ---------------------------------------------------------------------------
# Step 02 - Location
# ---------------------------------------------------------------------------
def _zip_field(state, city, street) -> str | None:
    """Render the ZIP control and return the selected code.

    The reference table resolves State + City + Street to exactly one ZIP for
    every recorded location, so the normal case is a read-only readout. The
    select branch exists so a future reference build with a genuine one-to-many
    address would still be handled without inventing a code.
    """
    options = ref.zips(state, city, street) if street else []

    html(c.label("ZIP Code"))
    if len(options) == 1:
        st.session_state["fl_zip"] = options[0]
        html(c.readout(options[0]))
        html(c.hint("Derived from the selected address. Not typed by hand."))
        return options[0]

    if len(options) > 1:
        st.selectbox(
            "ZIP Code", options, key="fl_zip", index=None,
            placeholder="Select ZIP Code", label_visibility="collapsed",
        )
        html(c.hint("This address is recorded with more than one ZIP Code."))
        return st.session_state.get("fl_zip")

    st.session_state.pop("fl_zip", None)
    html(c.readout(None))
    html(c.hint("Select a street to resolve the ZIP Code."))
    return None


def step_location() -> None:
    restore(*LOCATION_KEYS)
    html(c.panel_head(
        "Step 02",
        "Location",
        "The cardholder address. Each level is filtered by the one above it, so "
        "only combinations that exist in the research dataset can be selected.",
    ))

    state = st.session_state.get("fl_state")
    city = st.session_state.get("fl_city")
    street = st.session_state.get("fl_street")

    left, right = st.columns([1, 1], gap="medium")

    with left:
        html(c.label("State"))
        st.selectbox(
            "State", ref.states(), key="fl_state", index=None,
            placeholder="Select state", label_visibility="collapsed",
            on_change=on_state_change,
        )
        state = st.session_state.get("fl_state")

    with right:
        html(c.label("City"))
        st.selectbox(
            "City", ref.cities(state) if state else [], key="fl_city",
            index=None, placeholder="Select state first" if not state else "Select city",
            label_visibility="collapsed", disabled=not state,
            on_change=on_city_change,
        )
        city = st.session_state.get("fl_city")

    html(c.label("Street"))
    st.selectbox(
        "Street", ref.streets(state, city) if city else [], key="fl_street",
        index=None, placeholder="Select city first" if not city else "Select street",
        label_visibility="collapsed", disabled=not city,
        on_change=on_street_change,
    )
    street = st.session_state.get("fl_street")

    low, high = st.columns([1, 1], gap="medium")
    with low:
        zip_code = _zip_field(state, city, street)
    with high:
        coords = ref.resolve_coordinates(state or "", city or "", street or "",
                                         zip_code or "")
        html(c.label("Customer Coordinates", required=False))
        html(c.readout(f"{coords[0]:.4f}, {coords[1]:.4f}" if coords else None))
        html(c.hint("Looked up from the address. Latitude and longitude are "
                    "never typed in."))

    html(c.divider())
    remember(*LOCATION_KEYS)
    nav_row(
        step=2,
        next_label="Continue to Transaction",
        back_step=1,
        validator=lambda: val.validate_location(
            field("fl_state"),
            field("fl_city"),
            field("fl_street"),
            field("fl_zip"),
        ),
    )


# ---------------------------------------------------------------------------
# Step 03 - Transaction
# ---------------------------------------------------------------------------
def _time_field(key: str, caption: str, low: int, high: int) -> None:
    st.selectbox(
        caption, list(range(low, high + 1)), key=key, index=None,
        format_func=lambda v: f"{v:02d}", placeholder=caption,
        label_visibility="collapsed",
    )


def step_transaction() -> None:
    restore(*TRANSACTION_KEYS)
    html(c.panel_head(
        "Step 03",
        "Transaction",
        "The transaction record itself. Amount, category and the exact timestamp "
        "are all model features.",
    ))

    left, right = st.columns([1, 1], gap="medium")

    with left:
        html(c.label("Amount"))
        with st.container(key="fl_amount"):
            st.text_input(
                "Amount", key="fl_amount_raw", placeholder="0.00",
                label_visibility="collapsed", on_change=on_amount_change,
            )
        html(c.hint(f"Minimum ${spec.AMOUNT_MIN:.2f}. Thousands separators are "
                    "optional; the field reformats itself."))

    with right:
        html(c.label("Category"))
        st.selectbox(
            "Category", ref.categories(), key="fl_category", index=None,
            format_func=ref.category_label, placeholder="Select category",
            label_visibility="collapsed",
        )

    week_col, month_col, date_col = st.columns([1, 1, 1], gap="medium")
    with week_col:
        html(c.label("Week"))
        st.selectbox(
            "Week", spec.WEEKDAY_NAMES, key="fl_weekday", index=None,
            placeholder="Select day", label_visibility="collapsed",
        )
    with month_col:
        html(c.label("Month"))
        st.selectbox(
            "Month", spec.MONTH_NAMES, key="fl_month", index=None,
            placeholder="Select month", label_visibility="collapsed",
        )
    with date_col:
        html(c.label("Date"))
        st.selectbox(
            "Date", list(range(spec.DATE_MIN, spec.DATE_MAX + 1)), key="fl_date",
            index=None, format_func=lambda v: f"{v:02d}",
            placeholder="Select date", label_visibility="collapsed",
        )

    html(c.label("Time"))
    with st.container(key="fl_time"):
        hh, s1, mm, s2, ss = st.columns([6, 1, 6, 1, 6], gap="small")
        with hh:
            _time_field("fl_hour", "HH", spec.HOUR_MIN, spec.HOUR_MAX)
        with s1:
            html(c.time_separator())
        with mm:
            _time_field("fl_minute", "MM", spec.MINUTE_MIN, spec.MINUTE_MAX)
        with s2:
            html(c.time_separator())
        with ss:
            _time_field("fl_second", "SS", spec.SECOND_MIN, spec.SECOND_MAX)
    html(c.hint("HH:MM:SS. Hour, minute and second are three separate model "
                "features."))

    html(c.note(
        '<strong>Transaction distance is derived, not asked for.</strong> '
        'The research dataset records a merchant coordinate per transaction that '
        'is offset from the customer coordinate within roughly one degree in each '
        'direction, and a merchant name does not identify a fixed location. '
        'FraudLens therefore reproduces that measured offset from the fields you '
        'enter, using a seed derived from those fields so the same input always '
        'yields the same distance, and computes the great-circle distance in '
        'kilometres. The trained model itself learned this feature from the real '
        'recorded coordinates.'
    ))

    html(c.divider())
    remember(*TRANSACTION_KEYS)
    nav_row(
        step=3,
        next_label="Continue to Review",
        back_step=2,
        validator=lambda: val.validate_transaction(
            field("fl_amount_raw"),
            field("fl_category"),
            field("fl_weekday"),
            field("fl_month"),
            field("fl_date"),
            field("fl_hour"),
            field("fl_minute"),
            field("fl_second"),
        ),
    )


# ---------------------------------------------------------------------------
# Step 04 - Review
# ---------------------------------------------------------------------------
def collect_input() -> TransactionInput | None:
    """Assemble a :class:`TransactionInput` from session state.

    Returns ``None`` if anything required is still missing or the address no
    longer resolves, so the review and result steps can never score a partial
    record.
    """
    gender = field("fl_gender")
    age = val.clamp_age(field("fl_age"))
    job = field("fl_job")
    state = field("fl_state")
    city = field("fl_city")
    street = field("fl_street")
    zip_code = field("fl_zip")
    amount, amount_error = val.parse_amount(field("fl_amount_raw"))
    category = field("fl_category")
    weekday = field("fl_weekday")
    month = field("fl_month")
    date = field("fl_date")
    hour = field("fl_hour")
    minute = field("fl_minute")
    second = field("fl_second")

    coords = ref.resolve_coordinates(state or "", city or "", street or "",
                                     zip_code or "")
    required = (gender, age, job, state, city, street, zip_code, amount,
                category, weekday, month, date, hour, minute, second)
    if amount_error or coords is None or any(v is None or v == "" for v in required):
        return None

    return TransactionInput(
        gender=gender, age=int(age), job=job,
        state=state, city=city, street=street, zip_code=str(zip_code),
        lat=coords[0], long=coords[1],
        amount=float(amount), category=category, day_of_week=weekday,
        month_name=month, date=int(date), hour=int(hour),
        minute=int(minute), second=int(second),
    )


def review_groups(tx: TransactionInput, distance_km: float):
    return [
        ("Customer", [
            ("Gender", ref.gender_label(tx.gender), ""),
            ("Age", f"{tx.age}", ""),
            ("Job", tx.job, ""),
        ]),
        ("Location", [
            ("State", tx.state, ""),
            ("City", tx.city, ""),
            ("Street", tx.street, ""),
            ("ZIP Code", tx.zip_code, "mono"),
            ("Coordinates", f"{tx.lat:.4f}, {tx.long:.4f}", "mono"),
        ]),
        ("Transaction", [
            ("Amount", f"$ {val.format_amount(tx.amount)}", "mono"),
            ("Category", ref.category_label(tx.category), ""),
            ("Week", tx.day_of_week, ""),
            ("Month", tx.month_name, ""),
            ("Date", f"{tx.date:02d}", "mono"),
            ("Time", f"{tx.hour:02d}:{tx.minute:02d}:{tx.second:02d}", "mono"),
            ("Transaction distance", f"{distance_km:.2f} km", "derived"),
        ]),
    ]


def step_review() -> None:
    html(c.panel_head(
        "Step 04",
        "Review",
        "Everything the model will receive. Go back to any step to change a "
        "value; the transaction is only scored when you choose to run it.",
    ))

    tx = collect_input()
    if tx is None:
        html(c.errors(
            ["Some fields are still incomplete. Use Back to finish the earlier "
             "steps."],
            title="This transaction cannot be reviewed yet",
        ))
        html(c.divider())
        nav_row(step=4, next_label="Back to Transaction", back_step=1,
                validator=lambda: ["Complete the form before predicting."],
                next_step=3)
        return

    _, distance_km, _, _ = build_model_row(tx)
    html(c.review(review_groups(tx, distance_km)))
    html(c.hint("Transaction distance is derived from the address and the "
                "transaction fields, as described in the previous step."))

    html(c.divider())
    left, right = st.columns([1, 1], gap="small")
    if left.button("Back", key="fl_back_4", type="secondary", width="stretch"):
        goto(3)
        st.rerun()
    if right.button("Run prediction", key="fl_predict", type="primary",
                    width="stretch"):
        st.session_state["fl_prediction"] = predict(tx)
        goto(5)
        st.rerun()


# ---------------------------------------------------------------------------
# Step 05 - Result
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.50   # model.predict default; not a tuned risk band


def step_result() -> None:
    html(c.panel_head(
        "Step 05",
        "Result",
        "The model output for this transaction record.",
    ))

    prediction = st.session_state.get("fl_prediction")
    if prediction is None:
        html(c.errors(["No prediction has been produced yet."],
                      title="Nothing to show"))
        html(c.divider())
        if st.button("Back to Review", key="fl_back_5_empty", type="primary",
                     width="stretch"):
            goto(4)
            st.rerun()
        return

    row = prediction.model_row
    metrics = [
        ("Amount", f"$ {val.format_amount(row['amt'])}"),
        ("Transaction distance", f"{prediction.transaction_distance_km:.2f} km"),
        ("Category", ref.category_label(row["category"])),
        ("Time", f"{row['transaction_hour']:02d}:{row['transaction_min']:02d}:"
                 f"{row['transaction_seconds']:02d}"),
    ]
    html(c.verdict(
        is_fraud=prediction.is_fraud,
        probability=prediction.fraud_probability,
        threshold=DECISION_THRESHOLD,
        metrics=metrics,
    ))

    html(c.divider())
    left, right = st.columns([1, 1], gap="small")
    if left.button("Back to Review", key="fl_back_5", type="secondary",
                   width="stretch"):
        goto(4)
        st.rerun()
    if right.button("New transaction", key="fl_reset", type="primary",
                    width="stretch"):
        reset_all()
        st.rerun()


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
STEP_RENDERERS = {
    1: step_customer,
    2: step_location,
    3: step_transaction,
    4: step_review,
    5: step_result,
}


def main() -> None:
    theme.configure_page()
    theme.inject_css()
    init_state()

    try:
        _, _, _, metadata = load_artifacts()
    except (FileNotFoundError, ValueError) as exc:
        html(c.errors([str(exc)], title="The model artifacts could not be loaded"))
        return

    html(c.topbar(metadata["version"], metadata["n_features"]))
    html(c.stepper(st.session_state["fl_step"]))

    STEP_RENDERERS[st.session_state["fl_step"]]()

    html(c.footer(metadata["version"], metadata["created_utc"][:10]))


if __name__ == "__main__":
    main()










