"""Model training and comparison utilities."""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES_LEGACY = ["Elbow_Angle", "Shoulder_Angle", "Hip_Angle", "Knee_Angle"]
FEATURES_3D = [
    "Elbow_Flexion_3D",
    "Knee_Flexion_3D",
    "Hip_Flexion_3D",
    "Shoulder_Abduction_3D",
    "Shoulder_Flexion_3D",
    "Trunk_Inclination_3D",
    "Spine_Alignment_3D",
    "Pelvic_Tilt_3D",
]
TARGET = "Form_Label"


def _available_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for feat in FEATURES_LEGACY + FEATURES_3D:
        if feat in df.columns:
            cols.append(feat)
    return cols or FEATURES_LEGACY


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, StandardScaler, List[str]]:
    le = LabelEncoder()
    df = df.copy()
    df["Exercise_Encoded"] = le.fit_transform(df["Exercise"])
    feature_cols = _available_features(df)
    X = df[feature_cols + ["Exercise_Encoded"]].values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, le, scaler, feature_cols


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def train_gradient_boosting(X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X, y)
    return clf


def train_xgboost(X: np.ndarray, y: np.ndarray):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def evaluate_model(name: str, model, X_test, y_test, X_full, y_full) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring="accuracy")
    return {
        "name": name,
        "accuracy": accuracy,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "report": classification_report(y_test, y_pred, target_names=["Bad Form", "Good Form"]),
    }


def compare_models(X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, Any]], Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = [
        ("Random Forest", train_random_forest(X_train, y_train)),
        ("Gradient Boosting", train_gradient_boosting(X_train, y_train)),
    ]
    xgb = train_xgboost(X_train, y_train)
    if xgb is not None:
        candidates.append(("XGBoost", xgb))

    results = []
    best_model = candidates[0][1]
    best_acc = -1.0
    for name, model in candidates:
        result = evaluate_model(name, model, X_test, y_test, X, y)
        results.append(result)
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_model = model
    return results, best_model


def save_artifacts(
    model_dir: str,
    clf,
    le: LabelEncoder,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "form_classifier.pkl"), "wb") as handle:
        pickle.dump(clf, handle)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as handle:
        pickle.dump(le, handle)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as handle:
        pickle.dump(scaler, handle)
    with open(os.path.join(model_dir, "feature_columns.pkl"), "wb") as handle:
        pickle.dump(feature_cols, handle)


def write_comparison_report(results: List[Dict[str, Any]], report_path: str, dataset_rows: int) -> str:
    lines = [
        "=" * 72,
        "  BIOVISION AI — MODEL COMPARISON REPORT",
        "=" * 72,
        f"Dataset rows: {dataset_rows}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"Model: {result['name']}",
                f"  Test Accuracy : {result['accuracy'] * 100:.2f}%",
                f"  CV Accuracy   : {result['cv_mean'] * 100:.2f}% ± {result['cv_std'] * 100:.2f}%",
                result["report"],
                "",
            ]
        )
    best = max(results, key=lambda r: r["accuracy"])
    lines.append(f"Recommended model: {best['name']} ({best['accuracy'] * 100:.2f}% test accuracy)")
    lines.append("=" * 72)
    report = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    return report
