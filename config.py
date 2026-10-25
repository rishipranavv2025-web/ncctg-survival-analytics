from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
DATA_PATH = DATA_DIR / "lung.csv"
DATA_URL = "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/datasets/lung.csv"
FEATURES = ["age", "sex", "ph.ecog", "ph.karno", "pat.karno", "meal.cal", "wt.loss"]
DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "ph.ecog": "ECOG score",
    "ph.karno": "Physician Karnofsky",
    "pat.karno": "Patient Karnofsky",
    "meal.cal": "Meal calories",
    "wt.loss": "Weight loss",
}
