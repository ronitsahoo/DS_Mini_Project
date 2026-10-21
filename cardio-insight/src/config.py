"""Project-wide constants: paths, column groups, random seed and colour palette.

Every notebook and every app page imports from here, so a path or a colour only
has to be changed in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "heart.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42

# ---------------------------------------------------------------- columns
TARGET = "HeartDisease"
CONTINUOUS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = CONTINUOUS + CATEGORICAL

# In this dataset a value of 0 for these columns is physically impossible,
# so 0 means "not recorded" (a hidden missing value), not a real measurement.
ZERO_AS_MISSING = ["Cholesterol", "RestingBP"]

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Sex": "Sex",
    "ChestPainType": "Chest pain type",
    "RestingBP": "Resting blood pressure (mm Hg)",
    "Cholesterol": "Serum cholesterol (mg/dl)",
    "FastingBS": "Fasting blood sugar > 120 mg/dl",
    "RestingECG": "Resting ECG",
    "MaxHR": "Maximum heart rate (bpm)",
    "ExerciseAngina": "Exercise-induced angina",
    "Oldpeak": "Oldpeak – ST depression (mm)",
    "ST_Slope": "ST segment slope",
    "HeartDisease": "Heart disease",
}

CATEGORY_LABELS = {
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {"ASY": "Asymptomatic", "NAP": "Non-anginal pain",
                      "ATA": "Atypical angina", "TA": "Typical angina"},
    "FastingBS": {0: "No (≤ 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "RestingECG": {"Normal": "Normal", "ST": "ST-T wave abnormality",
                   "LVH": "Left ventricular hypertrophy"},
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}

# ---------------------------------------------------------------- palette
# Validated for colour-blind separation and contrast on white / dark navy surfaces.
NAVY = "#1B2A41"
TEAL = "#008C80"          # the single UI accent colour
PALETTE_LIGHT = ["#008C80", "#D4693F", "#3A68B5", "#B07D12"]
PALETTE_DARK = ["#1FA398", "#D96F45", "#5A86D0", "#B88A18"]
CLASS_COLORS = {0: PALETTE_LIGHT[0], 1: PALETTE_LIGHT[1]}   # no disease = teal, disease = coral
CLASS_NAMES = {0: "No heart disease", 1: "Heart disease"}
