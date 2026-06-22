#!/usr/bin/env python3
"""Build tabular dataset from exercise videos with 3D biomechanical features."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.dataset_builder import build_dataset
from src.utils.config import get_path, load_settings, resolve_data_dir
from src.utils.logging_config import setup_logging


def main() -> None:
    logger = setup_logging(__name__)
    settings = load_settings()
    base_dir = resolve_data_dir(settings)
    output_csv = str(get_path("dataset_csv", "gym_dataset.csv"))
    output_excel = str(get_path("dataset_excel", "gym_dataset.xlsx"))

    logger.info("Data directory: %s", base_dir)
    df = build_dataset(base_dir=base_dir, output_csv=output_csv, output_excel=output_excel)
    if df.empty:
        return
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(df.groupby(["Exercise", "Form"]).size().rename("Frames").to_string())
    print(f"\n  Total rows : {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
