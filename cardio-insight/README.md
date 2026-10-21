# CardioInsight – Heart Disease Risk Analysis

A syllabus-complete Data Science mini project: statistics you can explain, models validated properly, and a calm,
modern Streamlit app that runs entirely on your own machine.

**Author:** [YOUR NAME] · Roll no. [ROLL NO] · Class/Batch [BATCH] · Guide: [FACULTY NAME]
Department of Information Technology, VESIT Mumbai · Data Science Mini Project (CA2)

---

## Problem statement
Heart disease is the leading cause of death worldwide, and early detection saves lives. Using 11 routine clinical
measurements (age, blood pressure, cholesterol, ECG and exercise-test results, …) this project asks:

1. **Which clinical features differ significantly** between patients with and without heart disease?
2. **Can a machine learning model flag likely heart disease** while missing as few sick patients as possible (high recall)?
3. **What are the limits and ethical risks** of such a model?

## Dataset
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by
fedesoriano (Kaggle, 2021) – `data/heart.csv`.

* **918 patients × 12 columns** (11 features + target `HeartDisease`), merged from 5 hospital datasets
  (Cleveland, Hungary, Switzerland, Long Beach VA, Statlog).
* Continuous: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`
* Categorical: `Sex`, `ChestPainType`, `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope`
* Target balance: 508 with heart disease (55.3%) vs 410 without (44.7%)
* Hidden missing values: 172 rows with `Cholesterol = 0`, 1 row with `RestingBP = 0` (treated as missing)

## Syllabus mapping

| Module | Topic | Notebook | App page |
|---|---|---|---|
| 1 | Introduction, CRISP-DM, data understanding | `01_data_understanding` | Home |
| 2.1 | Mean, median, mode, weighted mean, range, quartiles, IQR, quartile deviation | `02_descriptive_statistics` | Data Explorer |
| 2.1 | Karl Pearson & Bowley skewness, kurtosis (lepto/meso/platy) | `02_descriptive_statistics` | Data Explorer |
| 2.1 | Karl Pearson correlation, covariance, pairplot, correlation ≠ causation | `03_correlation_analysis` | Correlations |
| 2.1 | Missing value imputation (mean, median, Linear/Ridge regression) | `06_missing_value_imputation` | — |
| 2.2 | Normality (Q-Q plots, Shapiro-Wilk) | `04_inferential_statistics` | — |
| 2.2 | Welch two-sample t-test, H0/H1, α, p-value | `04_inferential_statistics` | Statistical Tests → t-test |
| 2.2 | One-way ANOVA (2 groups and 3+ groups) | `04_inferential_statistics` | Statistical Tests → ANOVA |
| 2.2 | Chi-square test of independence | `04_inferential_statistics` | Statistical Tests → Chi-square |
| 2.2 | Confidence intervals | `04_inferential_statistics` | Statistical Tests → Confidence intervals |
| 2.2 | Type I / Type II errors, Central Limit Theorem | `04_inferential_statistics` | Model Performance (Type I/II card) |
| 3.1 | Hold-out, K-fold (5, 10), LOOCV, bootstrapping | `07_model_building_and_validation` | Model Performance |
| 3.1 | LR, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, XGBoost; confusion matrix, ROC | `07_model_building_and_validation` | Model Performance, Risk Predictor |
| 4 | Outliers: IQR, z-score, Isolation Forest, LOF | `05_outlier_detection` | Outliers |
| 6.2 | Ethics: bias, fairness (recall by sex), privacy, reproducibility | `08_ethics_and_limitations` | Ethics & About |

## Project structure
```
cardio-insight/
├── data/
│   ├── heart.csv                 # raw Kaggle data (never modified)
│   └── processed/                # cleaned / imputed data, statistics, outlier flags
├── notebooks/                    # 01 … 08, run in order
├── src/                          # reusable code shared by notebooks and app
│   ├── config.py                 # paths, columns, seed (42), colour palette
│   ├── data.py                   # loading + zeros → missing
│   ├── stats.py                  # hand-written descriptive formulas + hypothesis tests
│   ├── outliers.py               # IQR, z-score, Isolation Forest, LOF
│   ├── models.py                 # leakage-free pipelines, CV, LOOCV, bootstrap, explanations
│   └── plots.py                  # matplotlib (notebooks) + Plotly (app) styling
├── models/
│   ├── best_model.joblib         # complete fitted pipeline (preprocessing + SVM)
│   └── metrics.json              # every validation result, read by the app
├── app/                          # Streamlit multi-page app "CardioInsight"
│   ├── Home.py                   # entry point + navigation
│   ├── ui.py                     # theme CSS, cards, cached loaders
│   ├── views/                    # one file per page
│   └── assets/                   # logo
├── .streamlit/config.toml        # light & dark theme
├── reports/figures/              # all exported charts (PNG)
├── _original_backup/             # untouched copy of the original repository
├── requirements.txt              # pinned versions
├── LICENSE
└── README.md
```

## How to run locally
Everything runs on your own computer; nothing is deployed or uploaded.

**1. Create and activate a virtual environment** (Windows PowerShell, from the `cardio-insight` folder):
```bash
python -m venv .venv
```
```bash
.venv\Scripts\Activate.ps1
```
(macOS/Linux: `source .venv/bin/activate`)

**2. Install the pinned requirements:**
```bash
pip install -r requirements.txt
```

**3. Run the notebooks in order** (01 → 08). Either open them in Jupyter, or run all of them from the terminal:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```
Notebook 07 creates `models/best_model.joblib` and `models/metrics.json`; notebook 08 adds the fairness results.
(Both files are already included, so the app also works straight away.)

**4. Start the app:**
```bash
streamlit run app/Home.py
```
It opens at http://localhost:8501. The app follows your system's light/dark setting.

To put your own details in the app footer, edit the `AUTHOR` dictionary at the top of `app/ui.py`.

## App pages
1. **Home** – overview, KPI cards, how it works
2. **Data Explorer** – filters, dataset preview, descriptive statistics (both skewness coefficients), histogram/box plot
3. **Statistical Tests** – t-test, ANOVA, chi-square, confidence intervals with H0/H1, statistic, p-value, significance badge and a plain-English conclusion
4. **Correlations** – interactive heatmap (correlation/covariance), scatter matrix, pair view with trend line
5. **Outliers** – IQR / Z-score / Isolation Forest / LOF selector, flagged scatter, comparison and overlap tables
6. **Model Performance** – models × validation strategies, grouped bar chart, confusion matrix, ROC, Type I/II card, feature importance
7. **Risk Predictor** – two-column form, probability gauge with Low/Moderate/High bands, top contributing features, medical disclaimer
8. **Ethics & About** – dataset bias, recall by sex, privacy, reproducibility, credits

## Screenshots
_Add screenshots to `reports/screenshots/` and link them here._

| Home | Statistical Tests | Model Performance | Risk Predictor |
|---|---|---|---|
| ![Home](reports/screenshots/home.png) | ![Tests](reports/screenshots/tests.png) | ![Performance](reports/screenshots/performance.png) | ![Predictor](reports/screenshots/predictor.png) |

## Results summary
**Statistics**
* Heart-disease patients reach a much lower **MaxHR** (127.7 vs 148.2 bpm, Welch t-test p < 0.001) and are older (55.9 vs 50.6 years, p < 0.001).
* All six categorical features are significantly associated with heart disease (chi-square); the strongest are **ST_Slope** (Cramér's V 0.62), **ChestPainType** (0.54) and **ExerciseAngina** (0.49).
* Strongest linear correlations with the target: **Oldpeak (+0.40)** and **MaxHR (−0.40)**. Cholesterol is only weakly related (+0.10).
* Cholesterol is right-skewed and strongly leptokurtic (excess kurtosis 4.5); Oldpeak is the most skewed feature (Pearson +0.81).
* Missing cholesterol is not random (88% of those patients have heart disease), so rows are imputed rather than deleted; median imputation had the lowest error.

**Models** (stratified 10-fold CV, mean)

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| **SVM (selected)** | **0.910** | **0.885** | 0.924 |
| Random Forest | 0.906 | 0.884 | 0.928 |
| XGBoost | 0.900 | 0.883 | 0.926 |
| KNN | 0.898 | 0.875 | 0.914 |
| Logistic Regression | 0.890 | 0.877 | 0.920 |
| Decision Tree | 0.878 | 0.854 | 0.901 |
| Gaussian Naive Bayes | 0.866 | 0.869 | 0.913 |

* Selected by the highest (recall + F1) / 2 under 10-fold CV, because missing a sick patient (Type II error) is the dangerous mistake.
* On the untouched test set (184 patients) the SVM pipeline achieves **recall 0.902 (95% bootstrap CI 0.837–0.957)**, F1 0.885, ROC-AUC 0.941: 10 missed patients and 14 false alarms.
* 5-fold, 10-fold and LOOCV agree closely for Logistic Regression and Naive Bayes; the single hold-out split is noisier.
* **Fairness:** recall is 91.9% for men but 82.0% for women (10-fold CV), a gap of about 10 points, mainly because only 50 sick women are in the data.

## What I added to the original repository
The original repository (one large notebook, a single-page Streamlit app, a Flask app and a TensorFlow model) was restructured and extended:

* **Removed:** the deep learning (TensorFlow/Keras) model, the Flask app, the HTML templates and the `.idea/` folder, so that every method can be explained in a viva.
* **Restructured** into `data/`, `notebooks/`, `src/`, `models/`, `app/` and `reports/`, with a pinned `requirements.txt` and a local virtual environment.
* **8 new notebooks**, each with plain-English explanations before every code cell and a Key Findings section, covering CRISP-DM, descriptive statistics, correlation, inferential statistics, outliers, imputation, validation and ethics.
* **Hand-written statistics** (`src/stats.py`): mean, median, mode, weighted mean, quartiles, IQR, quartile deviation, Pearson and Bowley skewness, kurtosis and Pearson r, all cross-checked against pandas and scipy.
* **Inferential statistics:** Shapiro-Wilk, Q-Q plots, Welch t-tests, ANOVA (2 and 4 groups), chi-square with Cramér's V, confidence intervals, a CLT bootstrap demo, and Type I/II errors.
* **Outlier detection:** IQR, z-score, Isolation Forest and LOF, with comparison and overlap analysis.
* **Leakage-free sklearn Pipelines** (zeros → NaN → imputer → scaler/encoder → model) replacing the original refit-on-all-data preprocessing. Added KNN and Decision Tree.
* **Validation:** hold-out, stratified 5- and 10-fold CV, LOOCV, a bootstrap CI for recall, permutation importance, and a justified model-selection rule.
* **Ethics analysis:** sex and origin bias, recall by sex, privacy, and reproducibility checks.
* **A completely redesigned 8-page Streamlit app** with a custom light/dark theme, card layout, Plotly charts in a colour-blind-checked palette, a probability gauge and per-patient feature contributions.

## Acknowledgements
* **Starting point:** [nripstein/Heart-Disease-Prediction](https://github.com/nripstein/Heart-Disease-Prediction) by Noah Ripstein (MIT License). An untouched copy is kept in `_original_backup/`.
* **Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano (Kaggle), built from the UCI Heart Disease datasets (Cleveland, Hungary, Switzerland, Long Beach VA) and the Statlog (Heart) dataset.
* Libraries: pandas, NumPy, SciPy, scikit-learn, XGBoost, statsmodels, Matplotlib, Seaborn, Plotly, Streamlit.

## Medical disclaimer
CardioInsight is a student learning project. It is **not** a medical device and does not provide medical advice or a
diagnosis. Always consult a qualified doctor.

## License
MIT License – see [LICENSE](LICENSE). The original copyright notice (© 2023 Noah Ripstein) is retained as the MIT License requires.
