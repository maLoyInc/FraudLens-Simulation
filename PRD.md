# PRD — FraudLens
## Digital Transaction Fraud Detection System

**Document status:** Master Product Requirements Document  
**Version:** 1.0  
**Project:** FraudLens  
**Primary platform:** Streamlit backend with a highly customized HTML/CSS/JS presentation layer  
**Development approach:** One controlled step at a time; each step must be verified before the next step begins.

---

# 1. Product Overview

## 1.1 Product Name
**FraudLens**

## 1.2 Product Positioning
FraudLens is a web-based digital transaction fraud detection application that allows a user to enter customer, location, and transaction information and receive a machine-learning-based fraud prediction.

The application is intended for:
- academic / thesis demonstration,
- model evaluation demonstration,
- portfolio presentation,
- educational and awareness purposes.

FraudLens is **not** intended to make a final financial, banking, authorization, or legal decision.

## 1.3 Core Product Goal
Rebuild the current fraud detection prototype into a substantially more professional application with:
- a clean premium SaaS visual identity,
- a dark-only custom interface,
- a step-based transaction input experience,
- reliable and reproducible data preprocessing,
- a clean XGBoost training pipeline,
- a proper inference pipeline that does not retrain during runtime,
- accurate customer-location cascading,
- automatic transaction-distance calculation based on the validated Haversine methodology,
- clear prediction results,
- strong validation and testing,
- maintainable source code and documentation.

---

# 2. Current Project Baseline

The current legacy project is located in:

```text
app/
```

The legacy project contains the current Streamlit application, CSS files, model artifacts, notebooks, and cleaned dataset.

The legacy project is the **baseline**, not the final architecture.

Do not blindly preserve the current UI or preprocessing implementation.

The raw research dataset is located in:

```text
data/archive.zip
```

The raw dataset is the authoritative source for rebuilding and auditing the data pipeline.

The raw dataset must never be overwritten.

---

# 3. High-Level Product Direction

## 3.1 Visual Direction
FraudLens must use:
- **Dark-only theme**
- Premium SaaS visual style
- Fintech/security visual language
- Professional, minimal, trustworthy appearance
- Strong visual hierarchy
- Highly customized UI instead of default Streamlit-looking widgets wherever practical

Avoid:
- cyberpunk/neon styling,
- excessive gradients,
- excessive glassmorphism,
- dashboard clutter,
- excessive animation,
- fake AI branding,
- decorative elements that do not improve usability.

## 3.2 UI Technology Direction
The presentation layer should be approximately **95% custom UI** using:
- HTML where useful,
- CSS as the primary styling system,
- JavaScript where useful for interaction/input behavior,
- Streamlit primarily as the Python application/runtime/backend layer.

Native Streamlit components may still be used where they are the most reliable option, but the final application must not visually resemble an untouched/default Streamlit application.

The Streamlit toolbar/chrome should be minimized or hidden for the production presentation where technically safe, while preserving useful development functionality during local development.

---

# 4. Final Navigation / User Flow

FraudLens uses a **stepper** rather than one long flat form.

Target flow:

```text
FraudLens
   |
   v
Step 01 — Customer
   |
   v
Step 02 — Location
   |
   v
Step 03 — Transaction
   |
   v
Review / Predict
   |
   v
Prediction Result
```

The stepper must make progress clear without becoming visually heavy.

The user should not feel like they are filling out a raw machine-learning form.

---

# 5. Final Form Order

1. **Customer Information**
2. **Location Information**
3. **Transaction Details**

---

# 6. Customer Information

## 6.1 Fields

### Gender
- Required
- Keep the underlying model meaning unchanged.

### Age
- Display name: **Age**
- Required
- Valid range: **1–120**
- No visible `+` / `-` stepper controls.
- User can type the value.
- Values above 120 must be constrained/sanitized to 120 rather than accepted as invalid.
- Do not change model semantics.

### Job
- Required
- Keep the field meaning unchanged.
- Present as a searchable/usable selection control because the raw dataset contains many job categories.

---

# 7. Location Information

## 7.1 Final Order

```text
State
→ City
→ Street
→ ZIP Code
→ Transaction Distance
```

## 7.2 Cascading Location Requirements

Location selectors must be built from the actual raw dataset.

The system must implement:

```text
State
  ↓
City
  ↓
Street
  ↓
ZIP Code
```

Each subsequent selection must be filtered using the previous selections.

Example:

```text
State = California
↓
City options = only cities found in California
↓
City = Los Angeles
↓
Street options = only streets found in California + Los Angeles
↓
Street = ...
↓
ZIP options = only ZIP values valid for the selected State + City + Street
```

Do not hardcode location lists.

Do not use generic external location datasets to replace the research data.

Use the raw dataset as the authoritative lookup source unless a later audit proves otherwise.

## 7.3 Customer Coordinates

The customer location is represented by:
- `lat`
- `long`

The selected location must resolve the corresponding customer coordinates from the validated dataset mapping.

The user should not manually enter latitude/longitude unless a later verified design decision requires it.

## 7.4 ZIP Accuracy

ZIP values must be derived from the selected location combination and must not allow invalid combinations.

If multiple valid ZIP values exist for a selected location combination, do not silently invent or select an arbitrary ZIP.

---

# 8. City Population

`city_pop` must be **removed from the final model and application**.

Therefore:
- remove it from the final processed dataset,
- remove it from final preprocessing,
- remove it from the final model feature set,
- remove it from UI,
- retrain the model without it.

Do not merely hide it while leaving the old feature in the model.

The old artifacts are not the final artifacts after this change.

---

# 9. Transaction Distance

## 9.1 Goal

Transaction distance must be calculated automatically using the accepted Haversine methodology based on:
- customer latitude: `lat`
- customer longitude: `long`
- merchant latitude: `merch_lat`
- merchant longitude: `merch_long`

Formula:

```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
```

The implementation must be mathematically equivalent to the validated research method.

## 9.2 Important Data Constraint

Raw-data analysis has shown that `merchant` or `merchant + category` does **not** uniquely determine one merchant coordinate.

Therefore, do not assume:

```text
merchant name → permanent merchant coordinate
```

The method used to determine merchant coordinates for a new transaction must be researched and validated before final implementation.

This is an explicit open implementation/research item. Do not solve it with arbitrary averaging, arbitrary selection, or unsupported assumptions.

---

# 10. Transaction Details

## 10.1 Amount
Display name: **Amount**

Required.

Currency presentation:

```text
$ 34,234.00
```

The UI should display the US dollar symbol.

Visible `+` / `-` stepper controls must be removed.

Do not force an artificial dataset min/max into the visible UX unless the final validated product requirements require it.

## 10.2 Category
Current label `Merchant Category` becomes:

**Category**

Required.

Keep the underlying categorical meaning unchanged unless the final data audit determines otherwise.

## 10.3 Week
Current underlying feature `day_of_week` is displayed as:

**Week**

Readable values:

```text
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday
```

The underlying model representation must be handled consistently in preprocessing.

## 10.4 Month
UI representation:

```text
January
February
March
April
May
June
July
August
September
October
November
December
```

The final processed dataset should use month names as the categorical representation if validated as the best preprocessing choice during the data/model planning step.

The final model must be retrained using the final representation.

## 10.5 Date
Display name: **Date**

Required.

Use a dropdown/select-style control:

```text
01
02
03
...
31
```

No visible `+` / `-` controls.

## 10.6 Time
Raw timestamp data contains seconds.

The current preprocessing previously removed seconds, but FraudLens v2 must restore seconds as a model feature.

Final model time features:
- `transaction_hour`
- `transaction_min`
- `transaction_seconds`

Final UI representation:

```text
HH:MM:SS
```

Example:

```text
18:23:42
```

Time validation:
- Hour: `00–23`
- Minute: `00–59`
- Second: `00–59`

Invalid values must be prevented or corrected safely.

Do not use `24` as the maximum hour.

The UI should feel like a digital time input rather than three unrelated numeric widgets.

---

# 11. Final Feature Direction

The final model must not preserve the old 16-feature set unchanged.

At minimum:

### Remove
- `city_pop`

### Add
- `transaction_seconds`

### Rebuild/revalidate
- month representation,
- location-derived behavior,
- transaction_distance calculation.

The exact final feature list and order must be determined from the final preprocessing pipeline and saved metadata, not copied blindly from the legacy app.

Any additional feature retained from the legacy model must be explicitly verified during the data audit.

---

# 12. Customer Identity

Raw dataset fields include:
- `cc_num`
- `first`
- `last`

Treat `cc_num` as the stronger customer identifier for data analysis than `first + last`.

Do not use `first + last` as a unique customer key.

Customer coordinates:
- `lat`
- `long`

should be analyzed in relation to customer identity and location fields.

The final UI should not expose sensitive identifiers such as full card number unless explicitly required. In general, do not put `cc_num` into the end-user prediction form.

---

# 13. Merchant Semantics

`merchant` represents the merchant identifier/name associated with a transaction.

`category` represents the transaction category.

Do not interpret:
- `merchant = fraudster`
- `is_fraud = merchant is fraudulent`

The fraud target describes the transaction outcome/classification, not the legal status of the merchant.

Do not make causal or accusatory claims about merchants.

---

# 14. Machine Learning Rebuild

The final FraudLens model must be **retrained from the raw dataset**.

Do not simply keep using old model artifacts after changing the feature set.

Required logical stages:

```text
Raw Data
↓
Cleaning
↓
Feature Engineering
↓
Train/Test Separation
↓
Class Imbalance Handling
↓
Preprocessing
↓
XGBoost Training
↓
Cross Validation
↓
Evaluation
↓
Artifact Export
```

The exact sequence must be validated during the planning/audit step.

## 14.1 Class Imbalance
The existing research uses undersampling.

The final implementation must first audit that imbalance strategy and determine whether it should be preserved for methodological continuity.

Do not silently replace RandomUnderSampler with SMOTE or another strategy.

Any methodology change requires explicit review and justification.

## 14.2 Model
Primary algorithm:

**XGBoost**

Do not replace it with another primary classifier without explicit approval.

---

# 15. Model Evaluation

At minimum report:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Pay special attention to fraud:
- recall,
- precision,
- F1,
- ROC-AUC.

Do not rely on accuracy alone.

Compare the rebuilt model against the legacy model results to ensure the rebuild does not accidentally degrade the research objective.

---

# 16. Inference Architecture

Final runtime behavior:

```text
User Input
↓
Validation
↓
Feature Construction
↓
Preprocessing Artifact
↓
Model
↓
Prediction
↓
Probability
↓
UI Result
```

Runtime must never:
- retrain the model,
- fit the encoder,
- fit the scaler,
- overwrite model artifacts.

Training and runtime inference must be clearly separated.

---

# 17. Prediction Result

Final prediction UI should be a custom FraudLens result experience.

Minimum output:
- Prediction status
- Fraud probability
- clear safe/fraud state
- neutral explanation
- educational disclaimer

Do not introduce arbitrary:
- Low/Medium/High thresholds,
- custom risk scoring systems,
- unsupported feature explanations,
- fake causal explanations.

---

# 18. Validation UX

Required fields use a red asterisk:

```text
Amount *
Gender *
State *
```

Do not repeatedly write:
- `(Required)`
- `· Required`

Validation should not rely on large default Streamlit warning/error blocks.

Prefer:
- inline field feedback,
- step-level validation messages,
- clean custom error states,
- clear indication of incomplete fields.

---

# 19. Theme & Branding

FraudLens is **dark-only**.

Legacy:
```text
style_light.css
style_dark.css
```

are legacy styling and should not be treated as the final design.

The final application should use a unified custom dark design system.

Brand:
**FraudLens**

Suggested supporting descriptor:
**Digital Transaction Fraud Detection**

Final wording may be refined during UI implementation, but the product name should remain FraudLens unless explicitly changed.

---

# 20. Design System Direction

Target style:

**Premium SaaS + Fintech Security**

Characteristics:
- dark charcoal / navy base,
- controlled blue accent,
- restrained red for fraud state,
- restrained green for safe state,
- subtle borders,
- strong typography hierarchy,
- generous whitespace,
- compact but readable forms,
- professional data visualization,
- subtle motion where useful.

Avoid:
- excessive glow,
- neon cyberpunk,
- excessive shadows,
- excessive rounded cards,
- gimmicky AI visuals.

---

# 21. Typography

Use a modern professional sans-serif font strategy.

Requirements:
- strong numeric readability,
- clear heading hierarchy,
- readable field labels,
- muted helper text.

External web fonts are allowed only if deployment reliability and dependency implications are documented.

Final font choice is part of the UI implementation stage.

---

# 22. Streamlit Chrome

During local development, Streamlit development controls may remain visible.

For production presentation, minimize/hide unnecessary Streamlit chrome where technically safe.

Do not compromise:
- application functionality,
- error diagnosis,
- development workflow.

---

# 23. Data Safety & GitHub Hygiene

Raw dataset should remain separate from public application source.

Recommended conceptual separation:

```text
data/raw/
data/processed/
app/
training/
tests/
```

Do not commit raw source data publicly unless its licensing/usage terms explicitly allow it.

Avoid publishing unnecessary sensitive fields such as card numbers.

The final public repository should favor:
- cleaned/demo data where appropriate,
- reproducible preprocessing/training code,
- model metadata/artifacts as appropriate,
- README documentation,
- application source,
- tests.

---

# 24. Architecture Direction

Target maintainable architecture may evolve toward:

```text
fraudlens/
├── app/
│   ├── app.py
│   ├── inference/
│   ├── ui/
│   ├── assets/
│   └── models/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── training/
│   ├── notebooks/
│   └── scripts/
│
├── tests/
│
├── PRD.md
├── CLAUDE.md
└── requirements.txt
```

This is a direction, not a mandate to create every folder immediately.

Avoid over-engineering.

The architecture must remain understandable to a student and explainable during a thesis defense.

---

# 25. Legacy Files

Legacy files such as:

```text
style_light.css
style_dark.css
old .pkl artifacts
old cleaned CSV
legacy website.py
```

must not be deleted simply because they are obsolete.

A file may be removed only after its replacement is:
1. implemented,
2. tested,
3. verified,
4. and no longer needed.

Raw data must remain untouched.

---

# 26. Development Strategy

Development must be executed one step at a time.

Required pattern:

```text
STEP
↓
Implement only that STEP
↓
Run verification
↓
Report result
↓
Human review
↓
Next STEP
```

A step must not automatically trigger the next step.

No single prompt should attempt to complete the entire PRD.

---

# 27. Proposed Development Phases

## Phase 0 — Baseline Audit
Read-only audit of:
- legacy application,
- raw dataset archive,
- notebooks,
- model artifacts,
- requirements.

No code changes.

## Phase 1 — Final Data Specification
Lock:
- final features,
- transformations,
- customer identity strategy,
- location mappings,
- month representation,
- seconds,
- transaction distance strategy.

## Phase 2 — Data Preprocessing v2
Build reproducible preprocessing from raw data.

Raw data remains untouched.

## Phase 3 — Dataset Validation
Validate:
- row counts,
- missing values,
- category mappings,
- customer-location consistency,
- time extraction,
- distance calculation.

## Phase 4 — XGBoost Retraining
Train new model using final features and preprocessing.

## Phase 5 — Model Evaluation
Evaluate and compare against legacy metrics.

## Phase 6 — Final Model Artifacts
Export:
- encoder/preprocessor,
- scaler/transformer if needed,
- final XGBoost model,
- feature metadata,
- version metadata.

## Phase 7 — Inference Backend
Build clean runtime inference functions.

## Phase 8 — FraudLens UI Shell
Create the custom dark premium SaaS shell.

## Phase 9 — Customer Step
Implement customer inputs and validation.

## Phase 10 — Location Step
Implement:
- State → City → Street → ZIP,
- coordinate resolution.

## Phase 11 — Transaction Step
Implement:
- Amount,
- Category,
- Week,
- Month,
- Date,
- HH:MM:SS.

## Phase 12 — Merchant & Haversine
Implement and verify merchant-coordinate resolution and automatic distance calculation.

## Phase 13 — Prediction Result
Integrate prediction and probability into the custom UI.

## Phase 14 — Integration & Regression Testing
Run full application-level tests.

## Phase 15 — Production Polish
Finalize:
- documentation,
- repository structure,
- deployment configuration,
- Streamlit chrome handling,
- cleanup of legacy files.

---

# 28. Acceptance Criteria

## Data
- raw dataset remains untouched,
- final processed dataset is reproducible,
- `city_pop` is removed from the final model,
- `transaction_seconds` is part of the final feature pipeline,
- month representation is documented,
- customer location mapping is verified,
- merchant coordinate strategy is empirically justified,
- Haversine calculation is verified.

## ML
- XGBoost is retrained,
- train/test methodology is documented,
- imbalance handling is documented,
- metrics are reported,
- feature order is explicit,
- inference matches training preprocessing,
- runtime never retrains.

## UI
- dark-only custom SaaS interface,
- Customer → Location → Transaction stepper,
- State → City → Street → ZIP cascading location,
- Amount uses dollar presentation,
- no `+/-` controls on custom numeric inputs,
- Month uses readable names,
- Date uses dropdown,
- Time uses HH:MM:SS,
- required fields use red `*`,
- default Streamlit form appearance is substantially replaced.

## Reliability
- invalid inputs are blocked cleanly,
- no raw tracebacks in normal user interaction,
- artifact loading is robust,
- prediction works with valid data,
- safe and fraud states both render correctly,
- application runs from a clean environment.

---

# 29. Important Non-Goals

FraudLens v2 must not:
- become a banking transaction authorization system,
- accuse a merchant of being fraudulent solely because a transaction is labeled fraud,
- invent merchant coordinates,
- fabricate risk thresholds,
- fabricate model explanations,
- change methodology without evidence,
- use raw data as a runtime training dataset,
- retrain the model on app startup,
- become an over-engineered enterprise platform.

---

# 30. Decision Log / Open Items

The following require explicit verification during implementation:

1. Exact final merchant-coordinate resolution strategy for new transactions.
2. Exact final feature list/order after the new preprocessing audit.
3. Whether `merchant` remains a model feature and how it should be encoded.
4. Final month representation after evaluating modeling implications.
5. Final class-imbalance methodology for the retrained model.
6. Final typography/font implementation.
7. Production strategy for hiding/minimizing Streamlit chrome.
8. Final repository/public-data policy based on licensing and usage constraints.
9. Whether merchant should be a user-facing required field or an internally resolved field.

These decisions must be based on project evidence, not assumptions.

---

# 31. Engineering Rule

**The raw dataset is the source of truth for data facts.**

**The final training pipeline is the source of truth for final model behavior.**

**The final UI must accurately represent the actual model and data pipeline.**

No implementation should silently change one layer while leaving the other inconsistent.
