"""Model building, validation and prediction helpers.

Every model is wrapped in one sklearn Pipeline:

    zeros -> NaN  ->  imputer  ->  scaler / one-hot encoder  ->  classifier

Because imputation and scaling live *inside* the pipeline, they are re-fitted on
the training part of every split only. The test fold never leaks into the
preprocessing - this is what "avoiding data leakage" means.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve, make_scorer)
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import CATEGORICAL, CONTINUOUS, FEATURES, RANDOM_STATE, ZERO_AS_MISSING

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                "f1": "F1", "roc_auc": "ROC-AUC"}


# ---------------------------------------------------------------- preprocessing
def zero_to_nan(X):
    """Replace impossible zeros with NaN so the imputer can fill them (module-level so it can be pickled)."""
    X = np.asarray(X, dtype=float).copy()
    X[X == 0] = np.nan
    return X


def build_preprocessor():
    """ColumnTransformer that prepares every column type correctly."""
    other_num = [c for c in CONTINUOUS if c not in ZERO_AS_MISSING]
    zero_num_pipe = Pipeline([
        ("zero_to_nan", FunctionTransformer(zero_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("zero_num", zero_num_pipe, ZERO_AS_MISSING),
        ("num", num_pipe, other_num),
        ("cat", cat_pipe, CATEGORICAL),
    ])


def get_models():
    """The seven classifiers compared in this project (all with random_state=42 where applicable)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        "Gaussian Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=1),
    }


def make_pipeline(model):
    """Full pipeline = preprocessing + classifier."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def split_xy(df, target="HeartDisease"):
    return df[FEATURES], df[target]


# ---------------------------------------------------------------- model selection score
def selection_score(y_true, y_pred):
    """The project's model-selection criterion: the average of recall and F1."""
    return (recall_score(y_true, y_pred, zero_division=0) + f1_score(y_true, y_pred, zero_division=0)) / 2


SELECTION_SCORER = make_scorer(selection_score)

# Hyper-parameter grids used for GridSearchCV tuning in notebook 07
PARAM_GRIDS = {
    "SVM": {"model__C": [0.1, 0.5, 1, 2, 5, 10], "model__gamma": ["scale", 0.01, 0.03, 0.1]},
    "Random Forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 6, 10],
                      "model__min_samples_leaf": [1, 2, 4]},
    "XGBoost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.05, 0.1],
                "model__n_estimators": [200, 300]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "KNN": {"model__n_neighbors": [5, 7, 9, 11, 15, 21]},
    "Decision Tree": {"model__max_depth": [3, 4, 5, 6], "model__min_samples_leaf": [5, 10, 20]},
    "Gaussian Naive Bayes": {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
}


# ---------------------------------------------------------------- evaluation
def classification_metrics(y_true, y_pred, y_prob=None):
    """Accuracy, precision, recall, F1, specificity (+ ROC-AUC if probabilities are given)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def holdout_evaluate(pipe, X_train, X_test, y_train, y_test):
    """Fit on the training split, score on the untouched test split."""
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return classification_metrics(y_test, pred, prob), pred, prob


def kfold_evaluate(pipe, X, y, n_splits=5, random_state=RANDOM_STATE):
    """Stratified K-fold CV. Returns {metric: (mean, std)} over the folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std())) for m in SCORING}


def loocv_evaluate(pipe, X, y):
    """Leave-One-Out CV: n models, each tested on a single patient.

    Precision/recall cannot be computed on one patient, so we collect all n
    out-of-sample predictions and compute the metrics once on the whole set.
    """
    prob = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = classification_metrics(y, pred, prob)
    return {k: (float(m[k]), np.nan) for k in SCORING}


def bootstrap_ci(y_true, y_pred, metric=recall_score, n_boot=2000, confidence=0.95,
                 random_state=RANDOM_STATE):
    """Bootstrap confidence interval for a metric on a fixed test set.

    Resample the test patients with replacement many times, recompute the metric
    each time, and take the middle 95% of those values.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y_true[idx], y_pred[idx], zero_division=0)
    alpha = (1 - confidence) / 2
    return {"estimate": float(metric(y_true, y_pred)),
            "lower": float(np.quantile(scores, alpha)),
            "upper": float(np.quantile(scores, 1 - alpha)),
            "scores": scores}


def roc_points(y_true, y_prob, max_points=200):
    """ROC curve coordinates (down-sampled so they fit nicely in a JSON file)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}


# ---------------------------------------------------------------- prediction / explanation
def risk_band(p):
    """Translate a probability into a Low / Moderate / High band."""
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Moderate"
    return "High"


def reference_profile(df):
    """A 'typical patient': median of each continuous feature, mode of each categorical one."""
    ref = {}
    for c in CONTINUOUS:
        col = df[c].replace(0, np.nan) if c in ZERO_AS_MISSING else df[c]
        ref[c] = float(col.median())
    for c in CATEGORICAL:
        ref[c] = df[c].mode().iloc[0]
    return ref


def local_contributions(pipe, patient: pd.DataFrame, reference: dict):
    """How much does each feature push THIS patient's risk up or down?

    For each feature we replace the patient's value with the 'typical patient'
    value and see how the predicted probability changes:

        contribution = P(disease | patient) - P(disease | patient with that feature set to typical)

    Positive = this feature raises the risk; negative = it lowers it.
    The idea works for any model, so it does not depend on which model wins.
    """
    patient = patient[FEATURES]
    base = float(pipe.predict_proba(patient)[0, 1])
    variants = []
    for feat in FEATURES:
        v = patient.copy()
        v[feat] = reference[feat]
        variants.append(v)
    probs = pipe.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    contrib = pd.DataFrame({"feature": FEATURES, "value": patient.iloc[0].tolist(),
                            "contribution": base - probs})
    return base, contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------- persistence
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
