# BioVision AI

Local **biomechanical posture analysis** pipeline for gym exercises: record videos → extract MediaPipe joint angles → train a scikit-learn classifier → run **live webcam inference** with rep counting and form feedback.

**Stack:** Python 3.9+, OpenCV, MediaPipe Pose, NumPy/Pandas, scikit-learn (Random Forest). No deep-learning training step.

---

## Prerequisites

- **Python 3.9+** (tested on Apple Silicon; works on M-series Macs)
- **Webcam** for live inference
- Exercise videos stored locally under `Data/` (this folder is **not** in Git; see [Dataset layout](#dataset-layout))

---

## 1. Clone and enter the project

```bash
git clone https://github.com/hrshilll/Biovision-AI.git
cd Biovision-AI
```

---

## 2. Virtual environment and dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset layout (`Data/`)

Place **raw `.mp4` / `.mov` recordings** here. The extractor scans **immediate subfolders of `Data/`** as exercise names, each with **`good`** and **`bad`** subfolders (case-insensitive names, optional trailing spaces on folder names are handled).

Example:

```text
Data/
├── Bicep_curls/
│   ├── good/          # label 1 — correct form clips
│   └── bad/           # label 0 — incorrect form clips
├── Squats/
│   ├── good/
│   └── bad/
└── ...
```

Supported exercise folder names should align with live inference options, e.g. `Bicep_curls`, `Deadlifts`, `Planks`, `Pushups`, `Squats`.

**Configure the data root:** in `create_dataset.py`, set `BASE_DIR` to the **absolute path** of your `Data` directory (defaults in the repo may point to another machine).

```python
BASE_DIR = "/absolute/path/to/Biovision-AI/Data"
```

---

## 4. Build the tabular dataset (from videos)

From the project root (with `.venv` activated):

```bash
python create_dataset.py
```

**Outputs (project root):**

- `gym_dataset.csv` — rows for training / analysis  
- `gym_dataset.xlsx` — same data plus summary sheets  

Frames are skipped when pose is missing, landmarks have low visibility, or motion vs. the previous kept row is below a small threshold (see script constants). That is expected.

---

## 5. Train the classifier

Requires `gym_dataset.csv` from the previous step.

```bash
python train_model.py
```

**Writes to `models/`:**

- `form_classifier.pkl` — Random Forest  
- `scaler.pkl`, `label_encoder.pkl`  
- `training_report.txt` — metrics and classification report  

---

## 6. Live webcam inference

Requires the `models/*.pkl` files from training (or restore them from Git if you cloned a commit that includes them).

```bash
python live_inference.py
```

Follow the terminal prompt to choose an exercise. The window shows the skeleton, angles, **Good/Bad** prediction, rep count, and rule-based hints. **Space** or **Q** ends the session and saves CSV/XLSX under `session_results/`.

---

## 7. Optional: offline form report from CSV

Biomechanical rule checks per video in the dataset (Excel + text report):

```bash
python analyze_form.py
```

Reads `gym_dataset.csv`, writes `form_analysis_report.xlsx` and `form_analysis_report.txt`.

---

## 8. Optional: MediaPipe visibility smoke test

`debug_visibility.py` walks a local `bicep_curl/` tree for a quick pose-detection sanity check. Adjust paths inside the script if your layout differs.

---

## Project layout (high level)

| Path | Role |
|------|------|
| `create_dataset.py` | Video → angles → `gym_dataset.csv` / `.xlsx` |
| `train_model.py` | CSV → `models/*.pkl` |
| `live_inference.py` | Webcam + model + rep counting |
| `analyze_form.py` | CSV → rule-based analysis reports |
| `Data/` | **Local only** — exercise videos (ignored by Git) |
| `models/` | Trained artifacts + `training_report.txt` |
| `session_results/` | Exported live session tables |

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ModuleNotFoundError: openpyxl` | `pip install -r requirements.txt` |
| No rows in CSV | Check `BASE_DIR`, `good`/`bad` folder names, lighting, and full-body framing |
| Live script cannot load model | Run `train_model.py` first, or ensure `models/*.pkl` exist |
| macOS camera permission | System Settings → Privacy & Security → Camera → allow your terminal/IDE |

---

## License

Add a `LICENSE` file in the repository if you need a formal license for coursework or publication.
