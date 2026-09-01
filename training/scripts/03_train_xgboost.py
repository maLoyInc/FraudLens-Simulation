"""Train the FraudLens v2 XGBoost model and export inference artifacts.

Methodology is carried over from the research pipeline in
``App/dataset_training.ipynb`` so the rebuild stays comparable:

* ``train_test_split(test_size=0.2, random_state=101)``
* ``RandomUnderSampler(random_state=101)`` applied to the training split only
* ``OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)``
* ``StandardScaler``
* ``XGBClassifier(n_estimators=50, max_depth=7, random_state=101)``
* 5-fold ``cross_val_score`` on ROC-AUC

Two deliberate, documented differences from the legacy notebook:

1. The legacy notebook fitted the ordinal encoder on the whole dataset before
   splitting. Here it is fitted on the training split only, so no test-split
   information reaches the encoder.
2. The legacy notebook undersampled, wrote the balanced frame to CSV, then
   re-split *that balanced frame* into train and test - so its reported metrics
   were measured on a 50/50 test set. Here undersampling is applied to the
   training split only and the primary metrics are measured on the untouched,
   naturally imbalanced test split. The legacy-style balanced-test figure is
   also reported, clearly labelled, so the two can be compared.

Writes to ``models/v2/`` and ``training/reports/``. Never touches ``App/``.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fraudlens.core import config as cfg
from fraudlens.core import features as spec

RANDOM_STATE = 101
TEST_SIZE = 0.2
XGB_PARAMS = dict(n_estimators=50, max_depth=7, random_state=RANDOM_STATE)


def load_processed(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    fallback = path.with_suffix(".csv.gz")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise SystemExit(f"missing {path.name}; run training/scripts/02_build_processed.py first")


def evaluate(model, X_scaled, y_true, label: str) -> dict:
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "split": label,
        "n": int(len(y_true)),
        "n_fraud": int(np.sum(y_true)),
        "fraud_pct": round(float(np.mean(y_true) * 100), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_fraud": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_fraud": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_fraud": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_proba)), 6),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def main() -> None:
    spec.assert_feature_spec_consistent()
    cfg.MODEL_V2_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading processed training data ...", flush=True)
    df = load_processed(cfg.TRAIN_PROCESSED)
    X = df[spec.FEATURE_ORDER].copy()
    y = df[spec.TARGET].copy()
    del df

    # --- 1. Train / test separation on the raw class distribution -----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"train {X_train.shape} fraud={int(y_train.sum())} | "
          f"test {X_test.shape} fraud={int(y_test.sum())}")

    # --- 2. Ordinal encoding, fitted on the training split only ------------
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[spec.CATEGORICAL_FEATURES] = encoder.fit_transform(
        X_train[spec.CATEGORICAL_FEATURES]
    )
    X_test[spec.CATEGORICAL_FEATURES] = encoder.transform(
        X_test[spec.CATEGORICAL_FEATURES]
    )
    unseen = int((X_test[spec.CATEGORICAL_FEATURES] == -1).sum().sum())
    print(f"unseen categorical values in test split: {unseen}")

    # --- 3. Class imbalance: undersample the training split only ----------
    sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"resampled train {X_res.shape} "
          f"neg={int(len(y_res) - y_res.sum())} pos={int(y_res.sum())}")

    # --- 4. Scaling, fitted on the resampled training split ---------------
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res[spec.FEATURE_ORDER])
    X_test_scaled = scaler.transform(X_test[spec.FEATURE_ORDER])

    # --- 5. Train -------------------------------------------------------
    print("training XGBoost ...", flush=True)
    model = xgboost.XGBClassifier(**XGB_PARAMS)
    model.fit(X_res_scaled, y_res)

    # --- 6. Evaluate ----------------------------------------------------
    results = {"primary_imbalanced_test": evaluate(
        model, X_test_scaled, y_test.to_numpy(), "held-out test split (natural imbalance)"
    )}

    # Legacy-comparable view: balance the test split the same way the legacy
    # notebook implicitly did, so the two sets of numbers can be read together.
    X_test_bal, y_test_bal = RandomUnderSampler(
        random_state=RANDOM_STATE
    ).fit_resample(X_test, y_test)
    results["legacy_style_balanced_test"] = evaluate(
        model,
        scaler.transform(X_test_bal[spec.FEATURE_ORDER]),
        y_test_bal.to_numpy(),
        "held-out test split, undersampled 50/50 (legacy-comparable)",
    )

    # Fully unseen external holdout: the raw fraudTest.csv member.
    ext = load_processed(cfg.TEST_PROCESSED)
    X_ext = ext[spec.FEATURE_ORDER].copy()
    y_ext = ext[spec.TARGET].to_numpy()
    del ext
    X_ext[spec.CATEGORICAL_FEATURES] = encoder.transform(
        X_ext[spec.CATEGORICAL_FEATURES]
    )
    results["external_holdout_fraudTest"] = evaluate(
        model,
        scaler.transform(X_ext[spec.FEATURE_ORDER]),
        y_ext,
        "fraudTest.csv, never seen in training",
    )

    print("cross-validating (5-fold ROC-AUC on the resampled training set) ...", flush=True)
    cv = cross_val_score(
        xgboost.XGBClassifier(**XGB_PARAMS), X_res_scaled, y_res,
        scoring="roc_auc", cv=5,
    )
    results["cross_validation"] = {
        "scoring": "roc_auc",
        "folds": 5,
        "scores": [round(float(s), 6) for s in cv],
        "mean": round(float(cv.mean()), 6),
        "std": round(float(cv.std()), 6),
    }

    # --- 7. Export artifacts -------------------------------------------
    joblib.dump(encoder, cfg.ENCODER_PATH)
    joblib.dump(scaler, cfg.SCALER_PATH)
    joblib.dump(model, cfg.MODEL_PATH)

    importances = dict(
        sorted(
            zip(spec.FEATURE_ORDER, (round(float(v), 6) for v in model.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    metadata = {
        "name": "fraudlens-xgboost",
        "version": "v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "XGBClassifier",
        "xgb_params": XGB_PARAMS,
        "decision_rule": "model.predict, i.e. predict_proba >= 0.50",
        "feature_order": spec.FEATURE_ORDER,
        "n_features": len(spec.FEATURE_ORDER),
        "categorical_features": spec.CATEGORICAL_FEATURES,
        "numeric_features": spec.NUMERIC_FEATURES,
        "target": spec.TARGET,
        "changes_vs_legacy": {
            "removed": sorted(set(spec.LEGACY_FEATURE_ORDER) - set(spec.FEATURE_ORDER)),
            "added": sorted(set(spec.FEATURE_ORDER) - set(spec.LEGACY_FEATURE_ORDER)),
            "legacy_feature_order": spec.LEGACY_FEATURE_ORDER,
        },
        "month_representation": "integer 1-12; month names are presentation only",
        "day_of_week_representation": "English day name string, ordinal encoded",
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": False},
        "imbalance_handling": {
            "method": "RandomUnderSampler",
            "random_state": RANDOM_STATE,
            "applied_to": "training split only",
        },
        "encoder": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), fitted on training split",
        "scaler": "StandardScaler, fitted on the resampled training split",
        "unseen_categorical_values_in_test": unseen,
        "artifacts": {
            "encoder": cfg.ENCODER_PATH.name,
            "scaler": cfg.SCALER_PATH.name,
            "model": cfg.MODEL_PATH.name,
        },
        "feature_importances": importances,
        "metrics": results,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    cfg.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (cfg.REPORTS_DIR / "training_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n=== metrics ===")
    for key in ("primary_imbalanced_test", "legacy_style_balanced_test",
                "external_holdout_fraudTest"):
        r = results[key]
        print(f"{key}: n={r['n']:,} fraud={r['n_fraud']:,} "
              f"acc={r['accuracy']:.4f} P={r['precision_fraud']:.4f} "
              f"R={r['recall_fraud']:.4f} F1={r['f1_fraud']:.4f} "
              f"AUC={r['roc_auc']:.4f} AP={r['average_precision']:.4f} "
              f"CM={r['confusion_matrix']}")
    print(f"cv roc_auc: {results['cross_validation']['mean']:.6f} "
          f"+/- {results['cross_validation']['std']:.6f}")
    print(f"\nwrote artifacts to {cfg.MODEL_V2_DIR}")


if __name__ == "__main__":
    main()
