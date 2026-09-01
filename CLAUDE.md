# CLAUDE.md — FraudLens Engineering Rules

## 1. Purpose

FraudLens is a rebuild of the legacy fraud-detection app into a professional, dark-only, premium SaaS-style application.

The product requirements are defined in `PRD.md`.

`PRD.md` = what to build.  
`CLAUDE.md` = how Claude Code must work.

---

## 2. Work One Step at a Time

This is a strict incremental project.

For every requested step:

1. Inspect relevant files/data first.
2. Implement only the requested step.
3. Do not silently start later steps.
4. Run appropriate verification.
5. Report:
   - files changed,
   - what changed,
   - verification performed,
   - results,
   - remaining issues,
   - one recommended next step.
6. STOP.

Never implement the entire PRD in one operation.

The human reviewer decides when the next step begins.

---

## 3. Source of Truth

Use:

- `PRD.md` for product requirements.
- `data/archive.zip` for raw-data facts.
- `app/` for the legacy implementation and research baseline.
- The final training pipeline/artifact metadata for final model behavior.

Do not guess data relationships when they can be measured from the raw dataset.

## 3A. Project Instructions Protection

Do not modify `PRD.md` or `CLAUDE.md` unless the human explicitly requests an update to those documents.

Treat both files as project-level instructions and source-of-truth documents.

## 3B. STEP 0 Rule

STEP 0 is a strictly read-only baseline audit.

During STEP 0, Claude Code must not:
- create files,
- modify files,
- delete files,
- rename files,
- retrain models,
- transform raw data,
- change dependencies,
- change UI,
- change preprocessing.

STEP 0 exists only to understand and report the current state of the project and raw dataset.

---

## 4. Raw Dataset Protection

`data/archive.zip` is immutable.

Never:

- overwrite it,
- rewrite it,
- delete it,
- modify raw values in place,
- save cleaned data over it.

Create separate processed outputs.

Preferred future structure:

data/
├── raw/
└── processed/

## 5. Legacy Project Protection

app/ is the legacy baseline.

Do not immediately delete or overwrite:

legacy website.py,

style_light.css,

style_dark.css,

old .pkl artifacts,

legacy cleaned datasets,

notebooks.

Keep legacy material until its replacement is implemented, tested, and verified.

## 6. ML Integrity

Primary model:

XGBoost

Do not silently replace it.

Do not silently change:

target definition,

train/test methodology,

class-imbalance method,

feature engineering,

preprocessing,

decision threshold,

evaluation methodology.

Any material methodology change must be explicitly identified, justified, and reviewed.

## 7. Runtime Must Be Inference-Only

Final application code must never retrain at runtime.

Do not use runtime:

model.fit(...)
encoder.fit(...)
scaler.fit(...)
pickle.dump(...)
joblib.dump(...)
open(..., "wb")

Training and inference must be separate.

## 8. Feature Integrity

Never silently:

rename model features,

reorder model inputs without checking metadata,

add duplicate features,

drop model features without updating training,

add UI fields that are ignored by the model.

Whenever the final feature set changes:

update the training specification,

retrain if necessary,

export new artifacts,

verify feature names/order,

update inference,

run regression tests.

UI order may differ from model order, but inference must construct model input explicitly and correctly.

## 9. Dataset-Derived UI

Use validated project data for:

states,

cities,

streets,

ZIP codes,

jobs,

merchants,

other categorical options.

Do not hardcode long lists when the dataset can provide them.

Location cascade:

State
→ City
→ Street
→ ZIP

Every downstream selection must be valid for the selected upstream values.

Never invent invalid location combinations.

## 10. Merchant Coordinate Rule

Do NOT assume:

merchant → one fixed coordinate

Do NOT assume:

merchant + category → one fixed coordinate

Raw-data analysis has shown that merchant identifiers can occur with many merchant coordinate pairs.

Before implementing automatic merchant-coordinate resolution for a new transaction, inspect and validate the relationship empirically.

Never use an arbitrary:

first coordinate,

random coordinate,

average coordinate,

median coordinate,

nearest coordinate,

unless a specific step explicitly evaluates and approves that methodology.

## 11. Haversine Integrity

Transaction distance must use the approved Haversine methodology:

lat
long
merch_lat
merch_long

Expected unit:

kilometers

Do not silently change:

Earth radius,

coordinate order,

units,

radians conversion,

formula meaning.

Test the implementation against known examples.

## 11A. External Data Rule

Do not introduce external datasets, APIs, geocoding services, maps, merchant databases, or third-party location sources unless the human explicitly approves their use.

When the required information can be derived from the project's own raw dataset, prefer the project dataset.

Any external source must be documented before implementation.

## 12. Data Semantics

Treat:

merchant = merchant identifier/name associated with a transaction.

category = transaction category.

first / last = customer/cardholder names.

lat / long = customer-side coordinates.

merch_lat / merch_long = merchant-side coordinates for the observation.

is_fraud = transaction fraud target.

Never describe the merchant as "the fraudster" just because is_fraud = 1.

Do not make legal, causal, or accusatory claims about merchants or customers.

## 13. Customer Identity

Use cc_num as the stronger internal customer identifier than first + last.

Do not treat first + last as a unique customer key.

Do not expose raw cc_num in the normal user-facing prediction form.

## 14. Final Product Direction

Visual target:

Dark-only + Premium SaaS + Fintech Security

UI target:

~95% custom

Preferred presentation technologies:

HTML,

CSS,

JavaScript,

Streamlit as the Python runtime/backend.

The final UI should not look like an untouched/default Streamlit app.

## 15. UI Rules

Prefer:

dark charcoal/navy,

restrained blue accent,

restrained green safe state,

restrained red fraud state,

strong typography,

subtle borders,

generous whitespace,

responsive layout,

focused stepper.

Avoid:

cyberpunk/neon,

excessive gradients,

excessive glassmorphism,

excessive cards,

excessive rounded elements,

excessive animations,

fake AI visuals,

excessive emojis.

Final flow:

FraudLens
→ Customer
→ Location
→ Transaction
→ Review / Predict
→ Result

## 16. Required Inputs

Required fields use:

Field Name *

The asterisk is red.

Do not use:

(Required)
· Required

as the final convention.

Avoid large default Streamlit warning/error blocks for normal validation. Prefer clean inline or step-level feedback.

## 17. Input Rules

Amount

Display like:

"$ 34,234.00"

No visible +/- steppers.

Age

Range:

1–120

No visible +/- steppers.

Week

Display readable values:

Monday ... Sunday

Month

Prefer readable month names:

January ... December

The internal model representation must be explicit and documented.

Date

Use:

01 ... 31

as a clean dropdown/select control.

Time

Use:

HH:MM:SS

Valid ranges:

hour: 00–23

minute: 00–59

second: 00–59

Seconds must be a real model feature if the PRD requires it. 

## 18. Location UX

Use:

State → City → Street → ZIP

When a parent selection changes:

State changes
→ reset City, Street, ZIP

City changes
→ reset Street, ZIP

Street changes
→ reset ZIP

Never keep stale downstream selections.

Customer coordinates should resolve from the validated location mapping.

Do not ask the user to type latitude/longitude unless explicitly approved.

## 19. City Population

city_pop must be removed from the final application and final model.

Do not merely hide it.

It must be removed from:

processed data,

preprocessing,

feature list,

UI,

final model.

This requires a clean retraining pipeline.

## 20. Transaction Seconds

Seconds exist in the raw timestamp.

Final model direction:

transaction_hour
transaction_min
transaction_seconds

Do not add a cosmetic seconds field that the model ignores.

## 20A. Final Feature Set Is Not Yet Fully Locked

The following are already approved directions:

- `city_pop` will be removed.
- `transaction_seconds` will be added.

However, the complete final feature list and exact feature order are NOT considered final until the data audit and model specification steps are completed.

Do not retrain or finalize production artifacts based on an assumed feature list before that step is explicitly approved.

## 21. Model Evaluation

At minimum evaluate:

Accuracy,

Precision,

Recall,

F1-score,

ROC-AUC,

Confusion Matrix.

Pay particular attention to fraud-class precision, recall, and F1.

Compare the rebuilt model with the legacy result contextually.

## 22. Validation & Testing

Every behavior-changing step needs appropriate verification.

Examples:

Data

row count,

columns,

dtypes,

nulls,

duplicate behavior,

key uniqueness.

Model

feature count,

feature names/order,

preprocessing compatibility,

model loading,

inference.

UI

step transitions,

validation,

cascading reset behavior,

responsive behavior where practical.

Inference

valid input,

invalid input,

missing input,

preprocessing failure,

prediction output.

Never claim a visual/runtime test was performed if it was not actually run.

## 23. Decision Discipline

When something is ambiguous:

Inspect raw data.

Inspect legacy code/notebooks.

Quantify the ambiguity.

Present the evidence/options.

STOP if a product or methodology decision is required.

Do not pick a convenient value just to make the code run.

Hard-stop examples:

merchant-coordinate ambiguity,

feature add/remove decisions,

methodology changes,

class-imbalance changes,

new external data source,

licensing/legal assumptions,

destructive migration,

dependency changes with artifact compatibility risk.

## 24. No Silent Scope Expansion

Do not fix unrelated issues during a requested step.

If an unrelated issue is discovered:

record it,

leave it untouched,

address it in a later explicit step.

## 25. Code Quality

Prefer:

small focused functions,

clear names,

explicit transformations,

deterministic behavior,

testable logic,

comments explaining non-obvious technical decisions.

Avoid:

giant functions,

hidden side effects,

duplicated business logic,

unnecessary abstractions,

over-engineering.

The final code must remain understandable for a student and thesis examiner.

## 26. Comments & Documentation

Comments must describe the current implementation.

Do not include internal conversation/process labels such as:

STEP 3.5
STEP 6A report
Claude prompt
conversation instruction

Comments must make sense to future maintainers and examiners.

## 27. Artifact Versioning

When retraining creates new artifacts, do not immediately overwrite legacy artifacts.

Use a clear versioned strategy such as:

models/
├── legacy/
└── v2/

Keep legacy artifacts until the replacement has passed end-to-end verification.

## 28. Dependency Rules

Do not randomly upgrade:

Python,

Streamlit,

scikit-learn,

XGBoost,

pandas,

numpy.

When model serialization depends on specific versions, verify compatibility before changing versions.

Any dependency change that could affect model loading or behavior must be tested.

## 29. Git Hygiene

Do not commit:

.venv/
__pycache__/
*.pyc
temporary files
logs
debug dumps
secrets
API keys

Keep raw data separate from public source unless its licensing explicitly allows distribution.

## 30. Required Step Report

After each requested step, report:

A. Files Changed

Exact files modified.

B. What Changed

Concise summary.

C. Verification

Exact checks/tests performed.

D. Results

What passed or failed.

E. Remaining Issues

Only issues relevant to the step.

F. Recommended Next Step

Exactly one recommendation.

Then STOP.

## 31. Definition of Done

A step is complete only when:

requested changes are implemented,

appropriate verification is performed,

unrelated scope is untouched,

results are reported,

no known critical regression remains for that step.

"Looks good" is not enough.

## 32. Master Rules

Do not guess when the data can be inspected.

Do not change methodology silently.

Do not modify the raw dataset.

Do not retrain at runtime.

Do not fabricate coordinates, thresholds, explanations, or relationships.

Do one step at a time.

Verify every step.

Stop and ask when a major methodological decision is ambiguous.