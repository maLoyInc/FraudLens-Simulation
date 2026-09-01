# FraudLens

Dark-only, custom-UI fraud detection demo for credit-card transaction records.
Streamlit runs the Python and transports the widgets; everything visual is the
design system in `fraudlens/ui/static/fraudlens.css`.

The application is **inference only**. It loads the artifacts in `models/v2/`
and never fits, retrains, or serialises a model at runtime — enforced by a
static AST check in `tests/test_runtime_safety.py`.

> Educational and portfolio use only. FraudLens is not a banking authorisation
> system and must not be used for a financial, legal, or account decision.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
streamlit run fraudlens/app.py
```

The app needs `Data/processed/reference/` and `models/v2/` to exist. Both are
already committed as generated outputs; to rebuild them from the raw archive see
[Reproducing the pipeline](#reproducing-the-pipeline).

---

## The flow

```
FraudLens → 01 Customer → 02 Location → 03 Transaction → 04 Review → 05 Result
```

- **Customer** — gender, age (1–120), job. The card number is never asked for:
  `cc_num` is not a model feature.
- **Location** — State → City → Street, each filtered by the level above it.
  ZIP and the customer coordinates are *resolved* from the dataset, never typed.
  Changing a parent clears every downstream selection.
- **Transaction** — amount, category, weekday, month, date, and `HH:MM:SS`.
- **Review** — every value the model will receive, including the derived
  transaction distance.
- **Result** — the assigned class and the probability behind it, against the
  trained 0.50 decision rule. No invented risk tiers or bands.

---

## Layout

```
fraudlens/
├── app.py                  Streamlit entry point: the five steps and the shell
├── core/
│   ├── config.py           every path, plus the app title and tagline
│   ├── features.py         the model contract: 16 features, fixed order
│   ├── geo.py              Haversine (R = 6371 km) and coordinate resolution
│   ├── reference.py        dataset-derived option lists and the location cascade
│   └── validation.py       amount parsing, age clamping, per-step validators
├── inference/
│   └── predictor.py        artifact loading, row construction, prediction
└── ui/
    ├── components.py       HTML builders (all dynamic text HTML-escaped)
    ├── theme.py            page config and stylesheet injection
    └── static/fraudlens.css   the design system

training/
├── scripts/01_audit_raw.py       read-only audit of the raw archive
├── scripts/02_build_processed.py raw → Data/processed/ + reference tables
├── scripts/03_train_xgboost.py   trains and exports models/v2/
└── reports/                      audit, build, and metrics JSON

models/v2/     ordinal_encoder · standard_scaler · xgboost_fraud_model · metadata
App/           legacy baseline, kept untouched for comparison
tests/         97 tests
```

`core/` and `inference/` are framework-free; only `app.py` and `ui/` import
Streamlit.

---

## The model

XGBoost `XGBClassifier(n_estimators=50, max_depth=7, random_state=101)` over 16
features in a fixed order:

```
category, amt, gender, state, city, street, zip, job, age, day_of_week,
transaction_hour, transaction_min, transaction_seconds, transaction_date,
transaction_month, transaction_distance
```

Preprocessing is `OrdinalEncoder(handle_unknown="use_encoded_value",
unknown_value=-1)` over the seven categorical features, then `StandardScaler`
over all sixteen. `FEATURE_ORDER` in `core/features.py` is the single source of
truth; `assert_feature_spec_consistent()` runs at import and `load_artifacts()`
refuses to start if `model_metadata.json` disagrees with it.

Changes from the legacy model:

| | Legacy | v2 |
|---|---|---|
| `city_pop` | feature | removed from data, preprocessing, UI, and model |
| `transaction_seconds` | absent | real model feature |
| Undersampling | applied before the split | training split only |
| Encoder fit | whole dataset | training split only |
| Reported metrics | balanced test set | naturally imbalanced test set (balanced figure also reported) |

The last three are methodology corrections and are named explicitly in
`training/scripts/03_train_xgboost.py` and in `models/v2/model_metadata.json`.

### Measured results

Held-out test split at natural imbalance (259,335 rows, 0.58% fraud):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9972 |
| Average precision | 0.8168 |
| Accuracy | 0.9752 |
| Fraud recall | 0.9780 |
| Fraud precision | 0.1865 |
| Fraud F1 | 0.3133 |

On the same split undersampled 50/50 — the legacy-comparable view — accuracy is
0.9744 and fraud F1 is 0.9744. On `fraudTest.csv`, never seen in training,
ROC-AUC is 0.9956 with fraud recall 0.9613. Five-fold CV ROC-AUC is
0.996879 ± 0.000902. Full numbers, confusion matrices, and feature importances
are in `training/reports/training_metrics.json`.

Read the imbalanced-split precision honestly: at a 0.50 threshold the model
catches 1,468 of 1,501 frauds and raises 6,403 false positives. That is the
behaviour of an undersampled recall-oriented classifier, not a production
authorisation gate.

---

## Transaction distance

`transaction_distance` is the Haversine distance between the customer and
merchant coordinates, in kilometres at R = 6371 km.

The audit of the raw archive found that a merchant identifier does **not** carry
a location: 0 of 693 merchants and 0 of 700 merchant+category pairs map to a
single `(merch_lat, merch_long)`. The merchant coordinate is instead offset from
the customer coordinate roughly uniformly within ±1° on each axis
(`max |Δlat| = 0.999999`, `mean 0.500263`; `max |Δlon| = 0.999997`,
`mean 0.500337`), and the resulting distance carries almost no class signal
(median 78.23 km legitimate vs 77.93 km fraud).

So:

- **Training** uses the true recorded `merch_lat` / `merch_long`. Nothing is
  fabricated in the data the model learned from.
- **Inference** reproduces that measured generative process: the offset is drawn
  from the same ±1° uniform box using a seed derived with SHA-256 from the
  transaction's own fields, so the same input always yields the same distance.
  SHA-256 rather than `hash()` because `hash()` is salted per process.

The Review and Transaction steps state that the distance is derived. The
merchant name is not a model feature and is deliberately absent from the form.

---

## Reproducing the pipeline

`Data/archive.zip` is immutable; every script reads it and writes elsewhere.

```bash
python training/scripts/01_audit_raw.py
```

```bash
python training/scripts/02_build_processed.py
```

```bash
python training/scripts/03_train_xgboost.py
```

Step 1 writes `training/reports/raw_audit.json` and changes nothing else.
Step 2 writes the processed parquet files plus `locations.csv`,
`vocabulary.json`, and `merchant_offset.json`, and aborts if the
State+City+Street → ZIP cascade ever stops being unambiguous. Step 3 writes
`models/v2/`.

The reference vocabulary is built from the training member only, so every option
the form offers is a value the encoder actually saw.

---

## Verification

```bash
python -m pytest tests -q
```

97 tests, all passing:

- `test_geo.py` — Haversine against known reference distances, symmetry, the
  approved radius, determinism and bounds of the offset draw
- `test_features.py` — feature count, order, the categorical/numeric partition,
  `city_pop` removed, `transaction_seconds` present, saved metadata agreement
- `test_validation.py` — amount parsing and formatting, age clamping, every
  step validator
- `test_reference.py` — the cascade narrows, every recorded address resolves to
  exactly one ZIP and one coordinate, no hardcoded option lists
- `test_inference.py` — artifact loading, row order, determinism, unseen
  categorical values, the documented decision rule, immutable input
- `test_runtime_safety.py` — AST walk of `fraudlens/` rejecting `fit`, `dump`,
  file write modes, and training-only imports
- `test_app_flow.py` — the real Streamlit runtime via `AppTest`: step
  transitions, inline validation, cascade resets, derived ZIP and coordinates,
  amount reformatting, the full predict path, and the reset

The UI flow tests execute the actual app script, so step behaviour is exercised
rather than asserted by hand. Visual appearance has not been machine-verified;
review that in a browser.

---

## Dependencies

See `requirements.txt`. The versions differ from `App/requirements.txt` because
the legacy pins have no wheels for Python 3.14.6; the v2 artifacts are trained
in this environment, so no legacy pickle is ever loaded across versions. The
legacy artifacts under `App/` are untouched. Do not bump scikit-learn, xgboost,
numpy, or pandas without re-running step 3 and the test suite.
