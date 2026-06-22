#!/usr/bin/env python3
"""
BioVision AI — Live webcam inference with 3D biomechanics.

Usage:
  python live_inference.py
  python live_inference.py --3d-view
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.live_runner import main

if __name__ == "__main__":
    main()
