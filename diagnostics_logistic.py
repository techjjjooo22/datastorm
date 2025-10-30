from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

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


def collect_metrics(y_true, y_pred, y_proba):
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


def scan_thresholds(y_true, y_proba, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_metric = -np.inf
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_metric:
            best_metric = score
            best_threshold = threshold
    return best_threshold


def main():
    df = load_dataset(Path(DATA_PATH), SHEET_NAME)
    engineered = engineer_features(df)
    X, y = split_features_and_target(engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    C=0.1,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    metrics = collect_metrics(y_test, y_pred, y_proba)
    print("=== Logistic Regression Diagnostics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix (threshold 0.50):")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    threshold, cv_metrics = optimize_decision_threshold(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        metric="f1",
    )
    print(f"\nOptimal threshold (CV F1): {threshold:.3f}")
    for key, value in cv_metrics.items():
        print(f"cv_{key}: {value:.4f}")
    test_threshold_metrics, conf_thr = evaluate_with_threshold(
        model=pipeline,
        X=X_test,
        y=y_test,
        threshold=threshold,
    )
    print("\nTest metrics at optimal threshold:")
    for key, value in test_threshold_metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix (optimal threshold):")
    print(conf_thr)

    calibrator = CalibratedClassifierCV(
        estimator=Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        C=0.1,
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        method="isotonic",
        cv=3,
    )
    calibrator.fit(X_train, y_train)
    train_proba_cal = calibrator.predict_proba(X_train)[:, 1]
    calibration_threshold = scan_thresholds(y_train, train_proba_cal)
    test_proba_cal = calibrator.predict_proba(X_test)[:, 1]
    test_pred_cal = (test_proba_cal >= 0.5).astype(int)
    calibration_metrics = collect_metrics(y_test, test_pred_cal, test_proba_cal)
    print("\n=== Calibrated Logistic Regression (threshold 0.50) ===")
    for key, value in calibration_metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix (calibrated, 0.50):")
    print(confusion_matrix(y_test, test_pred_cal))

    calibration_test_metrics, conf_cal_thr = evaluate_with_threshold(
        model=calibrator,
        X=X_test,
        y=y_test,
        threshold=calibration_threshold,
    )
    print(f"\nCalibrated optimal threshold: {calibration_threshold:.3f}")
    for key, value in calibration_test_metrics.items():
        print(f"cal_{key}: {value:.4f}")
    print("Confusion matrix (calibrated optimal threshold):")
    print(conf_cal_thr)


if __name__ == "__main__":
    main()
