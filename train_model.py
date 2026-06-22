#!/usr/bin/env python3
"""Train form classifier with model comparison report."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trainer import (
    compare_models,
    load_dataset,
    prepare_data,
    save_artifacts,
    train_random_forest,
    write_comparison_report,
)
from src.utils.config import get_path
from src.utils.logging_config import setup_logging


def main() -> None:
    logger = setup_logging(__name__)
    input_csv = str(get_path("dataset_csv", "gym_dataset.csv"))
    model_dir = get_path("model_dir", "models")

    try:
        df = load_dataset(input_csv)
    except FileNotFoundError:
        logger.error("%s not found. Run create_dataset.py first.", input_csv)
        return

    if df.empty:
        logger.error("Dataset is empty.")
        return

    logger.info("Loaded %d rows from %s", len(df), input_csv)
    X, y, le, scaler, feature_cols = prepare_data(df)
    results, best_model = compare_models(X, y)
    comparison_path = model_dir / "model_comparison_report.txt"
    report = write_comparison_report(results, str(comparison_path), len(df))
    print(report)

    rf_model = train_random_forest(X, y)
    save_artifacts(str(model_dir), rf_model, le, scaler, feature_cols)

    training_report_path = model_dir / "training_report.txt"
    best = max(results, key=lambda r: r["accuracy"])
    with training_report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"BioVision AI Training Report\n"
            f"Dataset rows: {len(df)}\n"
            f"Features: {feature_cols}\n"
            f"Best comparison model: {best['name']} ({best['accuracy']*100:.2f}%)\n"
            f"Saved primary model: Random Forest\n"
        )

    logger.info("Model saved → %s/form_classifier.pkl", model_dir)
    logger.info("Comparison report → %s", comparison_path)


if __name__ == "__main__":
    main()
