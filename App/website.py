import streamlit as st
import pandas as pd 
import pickle
import os

# ==================== Theme Toggle & Load CSS ====================
theme = st.sidebar.radio("Theme", ["🌞Light", "🌚Dark"], horizontal=True)

if theme == "🌞Light":
    css_file = "style_light.css"
else:
    css_file = "style_dark.css"

with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================== Label Mapping ====================
column_labels = {
    "amt": "Amount",
    "gender": "Gender",
    "state": "State",
    "city": "City",
    "city_pop": "City Population",
    "job": "Job",
    "category": "Merchant Category",
    "street": "Street",
    "zip": "ZIP Code",
    "age": "Customer Age",
    "day_of_week": "Day of Week",
    "transaction_min": "Transaction Minute",
    "transaction_hour": "Transaction Hour",
    "transaction_date": "Transaction Date",
    "transaction_month": "Transaction Month",
    "transaction_distance": "Transaction Distance"
}

# ==================== Input Validation Config ====================
# Numeric ranges below are derived from domain/calendar constraints where
# they apply (hour, minute, date, month), and otherwise from a review of the
# actual min/max/distribution in fraudTrain_dataset_cleaned.csv, widened with
# a reasonable safety margin so the bounds don't reject legitimate values
# that simply weren't present in this particular training sample.
#
# Dataset reference (fraudTrain_dataset_cleaned.csv):
#   amt                  : min=1.00      max=12,788.07   (float, right-skewed)
#   zip                  : min=1,257     max=99,783       (int)
#   city_pop             : min=23        max=2,906,700    (int, heavily right-skewed; median=2,470)
#   age                  : min=14        max=96           (int)
#   transaction_min      : min=0         max=59           (int, fixed calendar range)
#   transaction_hour     : min=0         max=23           (int, fixed calendar range)
#   transaction_date     : min=1         max=31           (int, fixed calendar range)
#   transaction_month    : min=1         max=12            (int, fixed calendar range)
#   transaction_distance : min=0.74      max=146.52       (float, km; median=78.68)
#
# value=None (REQUIRED fields): renders an empty widget — no number is
# pre-filled, so a value that was never touched by the user cannot be
# mistaken for a real, meaningful input. st.number_input returns None while
# the field is empty, which the required-field check below treats as
# "not filled in" and blocks prediction on.
NUMERIC_INPUT_CONFIG = {
    # Required. Domain: a transaction amount must be positive. No natural
    # upper bound exists, so the max is a generous sanity ceiling (~4x the
    # dataset max) to catch obvious data-entry mistakes without blocking
    # large purchases.
    "amt": dict(min_value=0.01, max_value=50000.00, value=None, step=0.01, format="%.2f"),

    # Required. Domain: valid 5-digit US ZIP code range.
    "zip": dict(min_value=0, max_value=99950, value=None, step=1),

    # Required. Domain: realistic human age range.
    "age": dict(min_value=0, max_value=120, value=None, step=1),

    # Required. Fixed calendar constraints.
    "transaction_min": dict(min_value=0, max_value=59, value=None, step=1),
    "transaction_hour": dict(min_value=0, max_value=23, value=None, step=1),
    "transaction_date": dict(min_value=1, max_value=31, value=None, step=1),
    "transaction_month": dict(min_value=1, max_value=12, value=None, step=1),

    # Optional. Population cannot be negative; ceiling set well above the
    # dataset max to allow for larger cities not present in this sample.
    # Default uses the dataset median (robust to the heavy right-skew) since
    # this field is OPTIONAL and a user may legitimately leave it untouched.
    "city_pop": dict(min_value=0, max_value=10_000_000, value=2470, step=1),

    # Optional. Distance cannot be negative; ceiling set well above the
    # dataset max as a safety margin. Default uses the dataset median since
    # this field is OPTIONAL and a user may legitimately leave it untouched.
    "transaction_distance": dict(min_value=0.0, max_value=5000.0, value=78.68, step=0.01, format="%.2f"),
}

# Categorical fields that MUST be explicitly selected by the user before a
# prediction can be made. "job" is required: the model needs a real value
# for this feature, so it must not be silently imputed.
REQUIRED_CATEGORICAL_COLS = ["category", "gender", "state", "city", "street", "day_of_week", "job"]

# Numeric fields that MUST be explicitly filled in by the user (rendered
# with value=None above, so an untouched field is genuinely empty, not a
# look-alike default such as 0 or 1).
REQUIRED_NUMERIC_COLS = [
    "amt", "zip", "age",
    "transaction_min", "transaction_hour", "transaction_date", "transaction_month",
]

# ==================== Form Presentation Config (UI only) ====================
# Everything below is presentation-only: it controls how the 16 features are
# GROUPED and ORDERED on screen. It does NOT define which features the model
# uses -- X.columns (derived from the dataset, further down) remains the
# single source of truth for feature names, and the encoder/scaler/model
# still receive data built from that same set of 16 columns regardless of
# the order fields are drawn in here.
FORM_SECTIONS = {
    "Transaction Details": {
        "help": "Information about the transaction and when it occurred.",
        "fields": [
            "amt", "category", "day_of_week", "transaction_month",
            "transaction_date", "transaction_hour", "transaction_min",
        ],
    },
    "Customer Information": {
        "help": "Basic information about the customer.",
        "fields": ["gender", "age", "job"],
    },
    "Location Information": {
        "help": "Location-related details associated with the transaction.",
        "fields": ["state", "city", "street", "zip", "city_pop", "transaction_distance"],
    },
}

# Short, one-line helper text shown under a field only where the label alone
# might not be enough context. Fields not listed here rely on their label.
FIELD_HELP_TEXT = {
    "amt": "Transaction amount in USD.",
    "transaction_hour": "Hour of day, from 0 to 23.",
    "transaction_min": "Minute within the hour, from 0 to 59.",
    "city_pop": "Population of the transaction city.",
    "transaction_distance": "Distance between customer and merchant in kilometers.",
}

# Union of the two required-field lists above, used only to decide whether a
# field's on-screen label shows "Required" or "Optional". The validation
# logic in the prediction block still uses REQUIRED_CATEGORICAL_COLS and
# REQUIRED_NUMERIC_COLS directly and is unaffected by this set.
REQUIRED_FIELDS = set(REQUIRED_CATEGORICAL_COLS) | set(REQUIRED_NUMERIC_COLS)

# ==================== Title & Description ====================
st.title("💳 Fraud Transaction Detection App")
st.markdown("""
This application helps you predict whether a transaction is **safe** or **potentially fraudulent** using machine learning.
""")

# ==================== Sidebar Instructions ====================
with st.sidebar:
    st.header("📘 How to Use")
    st.markdown("""
1. Enter transaction details.
2. Click **Predict**.
3. Review the prediction probability.
""")

    st.markdown("**Model:** XGBoost")

    st.markdown("""
**Field guide**
- Required — Must be filled
- Optional — Can use default value
""")

    st.markdown("""
---
Educational tool only. This prediction should not be used as a final financial decision.
""")

# ==================== Load Dataset ====================
@st.cache_data
def load_data():
    path = "fraudTrain_dataset_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error("❌ Dataset not found.")
        return None

df = load_data()
if df is None:
    st.stop()

# ==================== Prepare Data (for UI only) ====================
target_col = 'is_fraud'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=[target_col])

# ==================== Load Official Trained Artifacts ====================
# Inference-only: encoder, scaler, and model are loaded from the artifacts
# produced by the training pipeline (see dataset_training.ipynb / xgb_modeling.ipynb).
# They are NOT fitted or overwritten here.
@st.cache_resource
def load_artifacts():
    artifact_files = {
        "encoder": "ordinal_encoder.pkl",
        "scaler": "fraud_scaler.pkl",
        "model": "xgboost_fraud_model.pkl",
    }
    artifacts = {}
    for name, path in artifact_files.items():
        try:
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"❌ Artifact '{path}' not found. Please make sure it exists in the app directory.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to load artifact '{path}': {e}")
            st.stop()
    return artifacts["encoder"], artifacts["scaler"], artifacts["model"]

encoder, scaler, model = load_artifacts()

# ==================== Prediction Form ====================
st.subheader("Transaction Information")
st.markdown("Fill in the details below, then click **Predict** to check this transaction.")

def render_field(col):
    """Render a single form field (selectbox or number_input) and return its
    value. Purely a UI helper: it does not change feature names, validation
    rules, or NUMERIC_INPUT_CONFIG -- it only decides label text/caption."""
    label = column_labels.get(col, col)
    tag = "Required" if col in REQUIRED_FIELDS else "Optional"
    display_label = f"{label} · {tag}"

    if col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        options.insert(0, "None")
        value = st.selectbox(display_label, options, index=0)
    else:
        cfg = NUMERIC_INPUT_CONFIG.get(col, dict(value=0.0))
        value = st.number_input(display_label, **cfg)

    if col in FIELD_HELP_TEXT:
        st.caption(FIELD_HELP_TEXT[col])

    return value


with st.form("fraud_form"):
    user_input = {}

    for section_name, section in FORM_SECTIONS.items():
        st.markdown(f"### {section_name}")
        st.caption(section["help"])

        fields = section["fields"]
        for i in range(0, len(fields), 2):
            pair = fields[i:i + 2]
            row_cols = st.columns(len(pair))
            for widget_col, col in zip(row_cols, pair):
                with widget_col:
                    user_input[col] = render_field(col)

    submitted = st.form_submit_button("🔍 Predict")

# ==================== Prediction Logic ====================
if submitted:
    # --- 1. Required field validation ---
    # Required categorical fields still at "None", and required numeric
    # fields left empty (value=None from the widget), both block prediction
    # here: the encoder, scaler, and model are not run on incomplete data.
    missing_required_categorical = [
        column_labels.get(col, col)
        for col in REQUIRED_CATEGORICAL_COLS
        if user_input.get(col) == "None"
    ]
    missing_required_numeric = [
        column_labels.get(col, col)
        for col in REQUIRED_NUMERIC_COLS
        if user_input.get(col) is None
    ]
    missing_required = missing_required_categorical + missing_required_numeric
    if missing_required:
        st.warning(
            "⚠️ Please fill in the following required field(s) before predicting: "
            + ", ".join(missing_required)
        )
        st.stop()

    input_df = pd.DataFrame([user_input])

    # --- 2. Feature consistency check ---
    # Confirm the input actually contains every column the encoder and
    # scaler expect before attempting to transform anything.
    missing_encoder_feats = [f for f in encoder.feature_names_in_ if f not in input_df.columns]
    missing_scaler_feats = [f for f in scaler.feature_names_in_ if f not in input_df.columns]
    if missing_encoder_feats or missing_scaler_feats:
        st.error(
            "❌ The submitted data is missing feature(s) required by the model "
            "and cannot be processed. Please refresh the page and try again."
        )
        st.stop()

    # --- 3. Preprocessing & prediction, with friendly error handling ---
    try:
        # Encode categorical columns using the exact column order the encoder was fitted on
        input_df[encoder.feature_names_in_] = encoder.transform(input_df[encoder.feature_names_in_])

        # Reorder all columns to match the exact order the scaler/model were fitted on
        input_scaled = scaler.transform(input_df[scaler.feature_names_in_])

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
    except Exception as e:
        st.error(
            "❌ Something went wrong while processing this transaction. "
            "Please check your input and try again."
        )
        st.caption(f"Technical detail: {type(e).__name__}: {e}")
        st.stop()

    # --- 4. Display result ---
    st.subheader("Prediction Result")

    # Guard only: predict_proba should already return a value in [0, 1] by
    # construction. This does not alter the probability's meaning or the
    # value shown to the user (prob is still displayed unclamped below) --
    # it only protects st.progress() from a pathological out-of-range float.
    progress_value = min(max(float(prob), 0.0), 1.0)

    if prediction == 1:
        st.error("🚨 Potentially fraudulent transaction")
    else:
        st.success("✅ Transaction appears safe")

    st.metric("Fraud Probability", f"{prob:.2%}")
    st.progress(progress_value)

    st.caption(
        "This probability represents the model's estimated likelihood that "
        "the transaction belongs to the fraud class."
    )
    st.caption(
        "Educational tool only. This prediction should not be used as a "
        "final financial decision."
    )
