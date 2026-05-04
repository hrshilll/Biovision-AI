"""
train_model.py
──────────────
Trains a Random Forest classifier on gym_dataset.csv and saves:
  - models/form_classifier.pkl   (the trained model)
  - models/label_encoder.pkl     (exercise name encoder)
  - models/scaler.pkl            (feature scaler)
  - models/training_report.txt   (accuracy + classification report)

Run this ONCE after create_dataset.py has generated gym_dataset.csv.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INPUT_CSV  = "gym_dataset.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["Elbow_Angle", "Shoulder_Angle", "Hip_Angle", "Knee_Angle"]
TARGET   = "Form_Label"


def main():
    # ── Load data ─────────────────────────────
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌  {INPUT_CSV} not found. Run create_dataset.py first.")
        return

    if df.empty:
        print("❌  Dataset is empty.")
        return

    print(f"📂  Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"    Exercises : {df['Exercise'].unique().tolist()}")
    print(f"    Good rows : {(df[TARGET] == 1).sum()}")
    print(f"    Bad rows  : {(df[TARGET] == 0).sum()}\n")

    # ── Encode exercise names ──────────────────
    le = LabelEncoder()
    df["Exercise_Encoded"] = le.fit_transform(df["Exercise"])

    # ── Features: 4 angles + exercise identity ─
    X = df[FEATURES + ["Exercise_Encoded"]].values
    y = df[TARGET].values

    # ── Scale features ─────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train Random Forest ────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",   # handles unequal good/bad counts
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────
    y_pred    = clf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

    report_lines = [
        "=" * 60,
        "  BIOVISION AI — MODEL TRAINING REPORT",
        "=" * 60,
        f"\nDataset rows   : {len(df)}",
        f"Train rows     : {len(X_train)}",
        f"Test rows      : {len(X_test)}",
        f"\nTest Accuracy  : {accuracy * 100:.2f}%",
        f"CV Accuracy    : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%",
        "\nClassification Report:",
        classification_report(y_test, y_pred, target_names=["Bad Form", "Good Form"]),
        "\nConfusion Matrix (rows=actual, cols=predicted):",
        "              Predicted Bad  Predicted Good",
        f"  Actual Bad  {confusion_matrix(y_test, y_pred)[0][0]:13d}  {confusion_matrix(y_test, y_pred)[0][1]:13d}",
        f"  Actual Good {confusion_matrix(y_test, y_pred)[1][0]:13d}  {confusion_matrix(y_test, y_pred)[1][1]:13d}",
        "\nFeature Importances:",
    ]

    feature_names = FEATURES + ["Exercise"]
    for name, imp in sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: -x[1]
    ):
        report_lines.append(f"  {name:20s}: {imp:.4f}")

    report_lines.append("=" * 60)
    report = "\n".join(report_lines)

    print(report)

    report_path = os.path.join(MODEL_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── Save model artefacts ───────────────────
    with open(os.path.join(MODEL_DIR, "form_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅  Model saved    → {MODEL_DIR}/form_classifier.pkl")
    print(f"✅  Encoder saved  → {MODEL_DIR}/label_encoder.pkl")
    print(f"✅  Scaler saved   → {MODEL_DIR}/scaler.pkl")
    print(f"📄  Report saved   → {report_path}")


if __name__ == "__main__":
    main()
