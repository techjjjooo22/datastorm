from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from main import (
    DATA_PATH,
    RANDOM_STATE,
    SHEET_NAME,
    build_preprocessor,
    engineer_features,
    evaluate_with_threshold,
    load_dataset,
    optimize_decision_threshold,
    split_features_and_target,
)


OUTPUT_DIR = Path("saved_models")
MODEL_PATH = OUTPUT_DIR / "logistic_pd_model.joblib"
METADATA_PATH = OUTPUT_DIR / "logistic_pd_model_metadata.json"
CALIBRATED_MODEL_PATH = OUTPUT_DIR / "logistic_pd_model_calibrated.joblib"


def build_logistic_pipeline(preprocessor) -> Pipeline:
    classifier = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def collect_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def scan_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_metric = -np.inf
    best_metrics: Dict[str, float] = {}

    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        metrics = {
            "f1": f1_score(y_true, preds),
            "balanced_accuracy": balanced_accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "accuracy": accuracy_score(y_true, preds),
        }
        score = metrics.get(metric, metrics["f1"])
        if score > best_metric:
            best_metric = score
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    best_metrics["average_precision"] = average_precision_score(y_true, y_proba)
    best_metrics["brier_score"] = brier_score_loss(y_true, y_proba)
    return best_threshold, best_metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH, SHEET_NAME)
    engineered = engineer_features(df)
    X, y = split_features_and_target(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = build_logistic_pipeline(preprocessor)
    param_grid = {"classifier__C": [0.05, 0.1, 0.2, 0.5, 1.0]}

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    holdout_metrics = collect_metrics(y_test, y_pred, y_proba)
    holdout_report = classification_report(y_test, y_pred, digits=4)
    holdout_confusion = confusion_matrix(y_test, y_pred).tolist()

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    threshold, cv_threshold_metrics = optimize_decision_threshold(
        model=best_pipeline,
        X=X_train,
        y=y_train,
        cv=cv_outer,
        metric="f1",
    )
    test_threshold_metrics, threshold_confusion = evaluate_with_threshold(
        model=best_pipeline,
        X=X_test,
        y=y_test,
        threshold=threshold,
    )

    print("=== Logistic Regression Hold-out Metrics ===")
    for metric, value in holdout_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nClassification report (hold-out):")
    print(holdout_report)
    print("Confusion matrix (hold-out):")
    print(np.array(holdout_confusion))

    print(f"\nOptimal threshold (CV, maximize F1): {threshold:.3f}")
    print("CV metrics at optimal threshold:")
    for metric, value in cv_threshold_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nHold-out metrics at optimal threshold:")
    for metric, value in test_threshold_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Confusion matrix (hold-out, optimal threshold):")
    print(np.array(threshold_confusion))

    print("\n=== Calibrated Logistic Regression (isotonic, 3-fold) ===")
    calibrator = CalibratedClassifierCV(
        estimator=clone(best_pipeline),
        method="isotonic",
        cv=3,
    )
    calibrator.fit(X_train, y_train)
    train_proba_cal = calibrator.predict_proba(X_train)[:, 1]
    calibration_threshold, calibration_cv_metrics = scan_thresholds(y_train, train_proba_cal)

    y_proba_cal = calibrator.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_proba_cal >= 0.5).astype(int)
    calibration_metrics = collect_metrics(y_test, y_pred_cal, y_proba_cal)
    calibration_report = classification_report(y_test, y_pred_cal, digits=4)
    calibration_confusion = confusion_matrix(y_test, y_pred_cal).tolist()

    print("Metrics at threshold 0.50:")
    for metric, value in calibration_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Confusion matrix (calibrated, threshold 0.50):")
    print(np.array(calibration_confusion))

    print(f"\nOptimal calibrated threshold (maximize train F1): {calibration_threshold:.3f}")
    print("Training metrics at calibrated threshold:")
    for metric, value in calibration_cv_metrics.items():
        print(f"{metric}: {value:.4f}")

    calibration_test_threshold_metrics, calibration_threshold_confusion = evaluate_with_threshold(
        model=calibrator,
        X=X_test,
        y=y_test,
        threshold=calibration_threshold,
    )
    print("\nHold-out metrics at calibrated optimal threshold:")
    for metric, value in calibration_test_threshold_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Confusion matrix (calibrated, optimal threshold):")
    print(calibration_threshold_confusion)

    # Retrain on full dataset with best hyperparameters
    final_preprocessor = build_preprocessor(X)
    final_pipeline = build_logistic_pipeline(final_preprocessor)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X, y)

    final_calibration_pipeline = build_logistic_pipeline(build_preprocessor(X))
    final_calibration_pipeline.set_params(**best_params)
    final_calibrator = CalibratedClassifierCV(
        estimator=final_calibration_pipeline,
        method="isotonic",
        cv=3,
    )
    final_calibrator.fit(X, y)

    timestamp = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    joblib.dump(final_pipeline, MODEL_PATH)
    joblib.dump(final_calibrator, CALIBRATED_MODEL_PATH)

    metadata = {
        "trained_at_utc": timestamp,
        "data_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
        "holdout_confusion_matrix": holdout_confusion,
        "holdout_classification_report": holdout_report,
        "optimal_threshold": float(threshold),
        "cv_threshold_metrics": cv_threshold_metrics,
        "holdout_threshold_metrics": test_threshold_metrics,
        "threshold_confusion_matrix": threshold_confusion.tolist(),
        "calibration": {
            "metrics_threshold_0_50": calibration_metrics,
            "confusion_matrix_threshold_0_50": calibration_confusion,
            "classification_report": calibration_report,
            "optimal_threshold": float(calibration_threshold),
            "training_metrics_at_optimal_threshold": calibration_cv_metrics,
            "holdout_metrics_at_optimal_threshold": calibration_test_threshold_metrics,
            "holdout_confusion_matrix_at_optimal_threshold": calibration_threshold_confusion.tolist(),
            "calibrated_model_path": str(CALIBRATED_MODEL_PATH.resolve()),
        },
        "model_path": str(MODEL_PATH.resolve()),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved pipeline to: {MODEL_PATH}")
    print(f"Saved calibrated pipeline to: {CALIBRATED_MODEL_PATH}")
    print(f"Metadata written to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
