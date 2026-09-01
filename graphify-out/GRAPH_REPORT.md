# Graph Report - FraudLens  (2026-08-26)

## Corpus Check
- Corpus is ~11,038 words - fits in a single context window. You may not need a graph.

## Summary
- 27 nodes · 26 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.92)
- Token cost: 8,642 input · 2,156 output

## Community Hubs (Navigation)
- Legacy App Implementation
- ML Model & Inference
- Product Requirements
- Data Quality Issues
- Dataset Architecture
- Dependencies

## God Nodes (most connected - your core abstractions)
1. `FraudLens Product Requirements` - 7 edges
2. `STEP 0 Baseline Audit Report` - 4 edges
3. `Raw Dataset Structure (archive.zip)` - 3 edges
4. `XGBoost Model Artifacts (pkl files)` - 3 edges
5. `load_data()` - 2 edges
6. `load_artifacts()` - 2 edges
7. `render_field()` - 2 edges
8. `XGBoost Fraud Detection Model` - 2 edges
9. `Haversine Distance Calculation` - 2 edges
10. `Location Cascade (State→City→Street→ZIP)` - 2 edges

## Surprising Connections (you probably didn't know these)
- `FraudLens Engineering Rules` --references--> `FraudLens Product Requirements`  [EXTRACTED]
  CLAUDE.md → PRD.md
- `Incremental Step-by-Step Development` --rationale_for--> `FraudLens Product Requirements`  [EXTRACTED]
  CLAUDE.md → PRD.md
- `XGBoost Model Artifacts (pkl files)` --implements--> `XGBoost Fraud Detection Model`  [INFERRED]
  App/README.md → PRD.md
- `Location Cascade (State→City→Street→ZIP)` --conceptually_related_to--> `Raw Dataset Structure (archive.zip)`  [INFERRED]
  PRD.md → Report1.md
- `Raw Dataset Immutability Rule` --rationale_for--> `Raw Dataset Structure (archive.zip)`  [INFERRED]
  CLAUDE.md → Report1.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FraudLens Project Governance Triangle** — prd_fraudlens, claude_engineering_rules, report1_baseline_audit [EXTRACTED 1.00]
- **Merchant Coordinate Resolution Problem** — prd_haversine_distance, claude_merchant_coordinate_rule, report1_merchant_coordinate_ambiguity [EXTRACTED 1.00]
- **Legacy to V2 Rebuild Architecture** — app_streamlit_inference, report1_legacy_ml_pipeline, prd_xgboost_model, prd_stepper_workflow, prd_dark_saas_design [INFERRED 0.85]

## Communities (6 total, 1 thin omitted)

### Community 0 - "Legacy App Implementation"
Cohesion: 0.29
Nodes (6): load_artifacts(), load_data(), Render a single form field (selectbox or number_input) and return its value.…, render_field(), cache_data, cache_resource

### Community 1 - "ML Model & Inference"
Cohesion: 0.33
Nodes (6): Legacy App README, Streamlit Inference-Only Application, XGBoost Model Artifacts (pkl files), Runtime Inference-Only Principle, XGBoost Fraud Detection Model, Legacy ML Pipeline Analysis

### Community 2 - "Product Requirements"
Cohesion: 0.40
Nodes (5): FraudLens Engineering Rules, Incremental Step-by-Step Development, Dark-Only Premium SaaS Design System, FraudLens Product Requirements, Multi-Step Stepper UI Workflow

### Community 3 - "Data Quality Issues"
Cohesion: 0.40
Nodes (5): Merchant Coordinate Ambiguity Rule, Haversine Distance Calculation, STEP 0 Baseline Audit Report, Customer Identity Findings (cc_num), Merchant Coordinate Ambiguity Finding

### Community 4 - "Dataset Architecture"
Cohesion: 0.67
Nodes (3): Raw Dataset Immutability Rule, Location Cascade (State→City→Street→ZIP), Raw Dataset Structure (archive.zip)

## Knowledge Gaps
- **5 isolated node(s):** `Multi-Step Stepper UI Workflow`, `Dark-Only Premium SaaS Design System`, `FraudLens Engineering Rules`, `Customer Identity Findings (cc_num)`, `Legacy App Dependencies`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FraudLens Product Requirements` connect `Product Requirements` to `ML Model & Inference`, `Data Quality Issues`, `Dataset Architecture`?**
  _High betweenness centrality (0.252) - this node is a cross-community bridge._
- **Why does `STEP 0 Baseline Audit Report` connect `Data Quality Issues` to `ML Model & Inference`, `Dataset Architecture`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Raw Dataset Structure (archive.zip)` (e.g. with `Raw Dataset Immutability Rule` and `Location Cascade (State→City→Street→ZIP)`) actually correct?**
  _`Raw Dataset Structure (archive.zip)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Multi-Step Stepper UI Workflow`, `Dark-Only Premium SaaS Design System`, `FraudLens Engineering Rules` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._