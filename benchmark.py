from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone


warnings.filterwarnings("ignore", category=UserWarning)


DATA_PATH = Path("moldova_npl.xlsx")
SHEET_NAME = "NPL_Data"
TARGET_COLUMN = "npl_target"
ID_COLUMNS = ["loan_id", "borrower_id"]
DATE_COLUMNS = ["origination_date", "maturity_date"]
RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    best_estimator: Pipeline
    best_params: Dict[str, object]
    cv_metrics: Dict[str, Tuple[float, float]]
    test_metrics: Dict[str, float]
    brier_score: float
    confusion_matrix: np.ndarray
    classification_report: str
    best_threshold: Optional[float] = None
    threshold_metrics: Optional[Dict[str, float]] = None
    test_threshold_metrics: Optional[Dict[str, float]] = None
    optimized_confusion_matrix: Optional[np.ndarray] = None
    calibrated_estimator: Optional[CalibratedClassifierCV] = None
    calibration_metrics: Optional[Dict[str, float]] = None
    calibration_report: Optional[str] = None
    calibration_confusion_matrix: Optional[np.ndarray] = None
    calibration_threshold: Optional[float] = None
    calibration_threshold_metrics: Optional[Dict[str, float]] = None


def load_dataset(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    return pd.read_excel(path, sheet_name=sheet_name)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    for date_col in DATE_COLUMNS:
        engineered[date_col] = pd.to_datetime(engineered[date_col])

    engineered["origination_year"] = engineered["origination_date"].dt.year
    engineered["origination_month"] = engineered["origination_date"].dt.month
    engineered["origination_quarter"] = engineered["origination_date"].dt.quarter
    engineered["origination_dayofyear"] = engineered["origination_date"].dt.dayofyear

    engineered["maturity_year"] = engineered["maturity_date"].dt.year
    engineered["maturity_month"] = engineered["maturity_date"].dt.month
    engineered["maturity_quarter"] = engineered["maturity_date"].dt.quarter
    engineered["maturity_dayofyear"] = engineered["maturity_date"].dt.dayofyear

    engineered["term_days"] = (engineered["maturity_date"] - engineered["origination_date"]).dt.days
    engineered["tenor_years"] = engineered["tenor_months"] / 12.0

    engineered["interest_policy_spread"] = (
        engineered["interest_rate_annual"] - engineered["policy_rate_at_origination"]
    )
    engineered["real_interest_rate"] = (
        engineered["interest_rate_annual"] - engineered["inflation_at_origination"]
    )

    # Credit behaviour proxies
    engineered["bureau_delinquency_score"] = (
        engineered["previous_dpd_30"] * 30 + engineered["previous_dpd_60"] * 60
    )
    engineered["total_previous_dpd"] = engineered["previous_dpd_30"] + engineered["previous_dpd_60"]

    # Capacity and affordability ratios
    income_safe = engineered["monthly_income_mdl"].replace(0, np.nan)
    installment_safe = engineered["installment_mdl"].replace(0, np.nan)

    engineered["installment_to_income"] = engineered["installment_mdl"] / income_safe
    engineered["loan_to_income"] = engineered["loan_amount_mdl"] / income_safe
    engineered["income_minus_installment"] = engineered["monthly_income_mdl"] - engineered["installment_mdl"]

    engineered["stress_buffer_index"] = (
        engineered["pti"] * (1 + 0.25 * engineered["fx_mismatch"] + 0.5 * engineered["shock_2022_2023"])
    )

    engineered["installment_to_loan_amount"] = engineered["installment_mdl"] / engineered["loan_amount_mdl"].replace(
        0, np.nan
    )
    engineered["loan_to_income_annualized"] = engineered["loan_to_income"] / engineered["tenor_years"].replace(
        0, np.nan
    )
    engineered["dti_minus_pti"] = engineered["dti"] - engineered["pti"]
    engineered["ltv_times_pti"] = engineered["ltv"] * engineered["pti"]

    # Log transformations to stabilize skewed distributions
    for col in ["loan_amount_mdl", "monthly_income_mdl", "installment_mdl", "loan_amount_ccy"]:
        engineered[f"log_{col}"] = np.log1p(engineered[col])
    engineered["log_pti"] = np.log1p(engineered["pti"])
    engineered["log_dti"] = np.log1p(engineered["dti"])

    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

    engineered.drop(columns=ID_COLUMNS, inplace=True)

    engineered = engineered.drop(columns=DATE_COLUMNS)

    return engineered


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = feature_frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def model_configurations(preprocessor: ColumnTransformer) -> List[Dict[str, object]]:
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    hgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.1,
                    max_leaf_nodes=63,
                    max_depth=None,
                    max_iter=350,
                    min_samples_leaf=15,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                    early_stopping="auto",
                ),
            ),
        ]
    )

    gb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    subsample=0.9,
                ),
            ),
        ]
    )

    stacking_classifier = StackingClassifier(
        estimators=[
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    C=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.08,
                    max_iter=400,
                    min_samples_leaf=20,
                    max_leaf_nodes=63,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    max_depth=18,
                    max_features="sqrt",
                    min_samples_leaf=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )

    stacking_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", stacking_classifier),
        ]
    )

    return [
        {
            "name": "Logistic Regression",
            "estimator": logistic_pipeline,
            "search_type": "grid",
            "search_params": {
                "classifier__C": [0.1, 0.5, 1.0, 2.0],
            },
            "use_sample_weight": False,
        },
        {
            "name": "Random Forest",
            "estimator": rf_pipeline,
            "search_type": "grid",
            "search_params": {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 18],
                "classifier__min_samples_leaf": [5, 15],
            },
            "use_sample_weight": False,
        },
        {
            "name": "HistGradientBoosting",
            "estimator": hgbm_pipeline,
            "search_type": "grid",
            "search_params": {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_leaf_nodes": [63, 127],
                "classifier__min_samples_leaf": [15],
                "classifier__l2_regularization": [0.0, 0.1],
            },
            "use_sample_weight": True,
        },
        {
            "name": "Gradient Boosting",
            "estimator": gb_pipeline,
            "search_type": "grid",
            "search_params": {
                "classifier__n_estimators": [200, 300],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3],
                "classifier__min_samples_leaf": [20, 40],
                "classifier__subsample": [0.8, 1.0],
            },
            "use_sample_weight": True,
        },
        {
            "name": "Stacking Ensemble",
            "estimator": stacking_pipeline,
            "search_type": "grid",
            "search_params": {
                "classifier__final_estimator__C": [0.5, 1.0, 2.0],
                "classifier__logistic__C": [0.3, 0.5, 1.0],
            },
            "use_sample_weight": True,
        },
    ]


def fine_tune_histgradientboosting(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    scoring: Dict[str, str],
    sample_weight: Optional[np.ndarray],
) -> ModelResult:
    hgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    random_state=RANDOM_STATE,
                    early_stopping="auto",
                ),
            ),
        ]
    )

    param_distributions = {
        "classifier__learning_rate": np.linspace(0.03, 0.12, 10),
        "classifier__max_leaf_nodes": [31, 63, 95, 127, 159],
        "classifier__min_samples_leaf": [10, 15, 20, 30, 50],
        "classifier__l2_regularization": [0.0, 0.05, 0.1, 0.2, 0.4],
        "classifier__max_iter": [300, 400, 500, 600],
        "classifier__max_depth": [None, 8, 12],
        "classifier__max_bins": [255],
    }

    return evaluate_model(
        name="HistGradientBoosting (Randomized)",
        estimator=hgbm_pipeline,
        search_type="random",
        search_params=param_distributions,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
        scoring=scoring,
        n_iter=25,
        sample_weight=sample_weight,
    )


def evaluate_model(
    name: str,
    estimator: Pipeline,
    search_type: str,
    search_params: Dict[str, List[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    scoring: Dict[str, str],
    n_iter: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> ModelResult:
    print(f"\n=== {name} ===")
    grid_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    fit_params: Dict[str, np.ndarray] = {}
    if sample_weight is not None:
        fit_params["classifier__sample_weight"] = sample_weight

    if search_type == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_params,
            n_iter=n_iter or 20,
            cv=grid_cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=2,
        )
        search_label = "RandomizedSearch"
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=search_params,
            cv=grid_cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )
        search_label = "GridSearch"

    search.fit(X_train, y_train, **fit_params)

    best_estimator = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC AUC ({search_label}): {search.best_score_:.4f}")

    cv_scores = cross_validate(
        best_estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        fit_params=fit_params if fit_params else None,
        return_train_score=False,
        error_score="raise",
    )

    cv_metrics = {
        metric.replace("test_", ""): (np.mean(values), np.std(values))
        for metric, values in cv_scores.items()
        if metric.startswith("test_")
    }

    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]

    test_metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    brier = brier_score_loss(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)

    return ModelResult(
        name=name,
        best_estimator=best_estimator,
        best_params=search.best_params_,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        brier_score=brier,
        confusion_matrix=conf_matrix,
        classification_report=class_report,
    )


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "Model": result.name,
            "CV ROC AUC": result.cv_metrics["roc_auc"][0],
            "CV PR AUC": result.cv_metrics["average_precision"][0],
            "CV Accuracy": result.cv_metrics["accuracy"][0],
            "Test ROC AUC": result.test_metrics["roc_auc"],
            "Test PR AUC": result.test_metrics["average_precision"],
            "Test Accuracy": result.test_metrics["accuracy"],
            "Test Balanced Acc": result.test_metrics["balanced_accuracy"],
            "Test F1": result.test_metrics["f1"],
            "Test Brier": result.brier_score,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(by="Test ROC AUC", ascending=False).reset_index(drop=True)


def optimize_decision_threshold(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    sample_weight: Optional[np.ndarray] = None,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    oof_probas = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        estimator = clone(model)
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        fit_params: Dict[str, np.ndarray] = {}
        if sample_weight is not None:
            fit_params["classifier__sample_weight"] = sample_weight[train_idx]
        estimator.fit(X_train_fold, y_train_fold, **fit_params)
        oof_probas[valid_idx] = estimator.predict_proba(X_valid_fold)[:, 1]

    best_threshold = 0.5
    best_metric_value = -np.inf
    best_metrics: Dict[str, float] = {}

    for threshold in thresholds:
        preds = (oof_probas >= threshold).astype(int)
        metrics = {
            "f1": f1_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "recall": recall_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "accuracy": accuracy_score(y, preds),
        }
        metric_value = metrics.get(metric, metrics["f1"])
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["roc_auc"] = roc_auc_score(y, oof_probas)
    best_metrics["average_precision"] = average_precision_score(y, oof_probas)
    best_metrics["brier_score"] = brier_score_loss(y, oof_probas)

    return best_threshold, best_metrics


def evaluate_with_threshold(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "average_precision": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "brier_score": brier_score_loss(y, proba),
    }
    return metrics, confusion_matrix(y, preds)


def calibrate_model(
    model_result: ModelResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    print("\nCalibrating best model with isotonic regression (3-fold CV)...")
    calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    calibrator = CalibratedClassifierCV(
        estimator=clone(model_result.best_estimator),
        method="isotonic",
        cv=calibration_cv,
    )
    calibrator.fit(X_train, y_train)

    train_proba = calibrator.predict_proba(X_train)[:, 1]
    y_proba = calibrator.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred)

    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_metric = -np.inf
    best_cv_metrics: Dict[str, float] = {}
    for threshold in thresholds:
        preds = (train_proba >= threshold).astype(int)
        metrics_train = {
            "f1": f1_score(y_train, preds),
            "balanced_accuracy": balanced_accuracy_score(y_train, preds),
            "precision": precision_score(y_train, preds, zero_division=0),
            "recall": recall_score(y_train, preds, zero_division=0),
            "accuracy": accuracy_score(y_train, preds),
        }
        score = metrics_train["f1"]
        if score > best_metric:
            best_metric = score
            best_threshold = threshold
            best_cv_metrics = metrics_train

    cal_threshold = best_threshold
    best_cv_metrics["roc_auc"] = roc_auc_score(y_train, train_proba)
    best_cv_metrics["average_precision"] = average_precision_score(y_train, train_proba)
    best_cv_metrics["brier_score"] = brier_score_loss(y_train, train_proba)

    cal_test_metrics, cal_test_conf = evaluate_with_threshold(
        model=calibrator,
        X=X_test,
        y=y_test,
        threshold=cal_threshold,
    )

    model_result.calibrated_estimator = calibrator
    model_result.calibration_metrics = metrics
    model_result.calibration_report = report
    model_result.calibration_confusion_matrix = conf_matrix
    model_result.calibration_threshold = cal_threshold
    model_result.calibration_threshold_metrics = {
        "cv": best_cv_metrics,
        "test": cal_test_metrics,
        "confusion_matrix": cal_test_conf,
    }

    print("Calibration metrics at threshold 0.50:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix (calibrated, threshold 0.50):")
    print(conf_matrix)
    print(f"\nOptimal calibrated threshold (maximize CV F1): {cal_threshold:.3f}")
    print("Calibrated CV metrics at optimal threshold:")
    for key, value in best_cv_metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nCalibrated hold-out metrics at optimal threshold:")
    for key, value in cal_test_metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix (calibrated, optimal threshold):")
    print(cal_test_conf)

def display_feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.Series:
    print("\nComputing permutation importances on the hold-out set...")
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    feature_names = X.columns
    importances = pd.Series(result.importances_mean, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)
    return top_features


def main() -> None:
    df = load_dataset(DATA_PATH, SHEET_NAME)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("target_proportion"))

    engineered = engineer_features(df)
    X, y = split_features_and_target(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)
    configs = model_configurations(preprocessor)
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    results: List[ModelResult] = []
    for config in configs:
        sample_weight = sample_weight_train if config.get("use_sample_weight", False) else None
        result = evaluate_model(
            name=config["name"],
            estimator=config["estimator"],
            search_type=config["search_type"],
            search_params=config["search_params"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            scoring=scoring,
            n_iter=config.get("n_iter"),
            sample_weight=sample_weight,
        )
        results.append(result)

    print("\nRefining HistGradientBoosting with a broader randomized search (may take a couple of minutes)...")
    refined_result = fine_tune_histgradientboosting(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
        scoring=scoring,
        sample_weight=sample_weight_train,
    )
    results.append(refined_result)

    summary = summarize_results(results)
    print("\n=== Model Comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    def selection_score(res: ModelResult) -> float:
        return res.test_metrics["roc_auc"] - 0.5 * res.brier_score

    best_model = max(results, key=selection_score)

    print(f"\nBest model (ROC AUC - 0.5 * Brier criterion): {best_model.name}")
    print(f"Best params: {best_model.best_params}")
    print("\nClassification report (Test set):")
    print(best_model.classification_report)
    print("Confusion matrix (Test set):")
    print(best_model.confusion_matrix)
    print(f"Brier score (Test set): {best_model.brier_score:.4f}")

    best_threshold, threshold_metrics = optimize_decision_threshold(
        model=best_model.best_estimator,
        X=X_train,
        y=y_train,
        cv=cv,
        sample_weight=sample_weight_train,
        metric="f1",
    )
    best_model.best_threshold = best_threshold
    best_model.threshold_metrics = threshold_metrics

    test_threshold_metrics, optimized_conf_matrix = evaluate_with_threshold(
        model=best_model.best_estimator,
        X=X_test,
        y=y_test,
        threshold=best_threshold,
    )
    best_model.test_threshold_metrics = test_threshold_metrics
    best_model.optimized_confusion_matrix = optimized_conf_matrix

    print(f"\nOptimal decision threshold (maximizing CV F1): {best_threshold:.3f}")
    print("Cross-validated metrics at optimal threshold:")
    for key in ["f1", "balanced_accuracy", "accuracy", "precision", "recall", "roc_auc", "average_precision", "brier_score"]:
        if key in threshold_metrics:
            print(f"{key}: {threshold_metrics[key]:.4f}")

    print("\nTest metrics at optimal threshold:")
    for key in ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "brier_score"]:
        if key in test_threshold_metrics:
            print(f"{key}: {test_threshold_metrics[key]:.4f}")
    print("Confusion matrix (Test set, optimized threshold):")
    print(optimized_conf_matrix)

    calibrate_model(
        model_result=best_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    top_features = display_feature_importance(best_model.best_estimator, X_test, y_test)
    print("\nTop permutation importances (Test set):")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
