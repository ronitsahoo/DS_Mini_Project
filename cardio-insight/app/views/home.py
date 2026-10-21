import streamlit as st

import ui
from src import config as C

ui.require_data()
df = ui.load_raw()
metrics = ui.load_metrics()

ui.html("""
<div class="ci-hero">
  <div class="ci-eyebrow">Data Science Mini Project</div>
  <h1 class="ci-title">CardioInsight</h1>
  <p class="ci-subtitle" style="margin-bottom:0">Explore, test and model clinical data from 918 patients to understand
  and estimate the risk of heart disease - with statistics you can explain and a model you can trust.</p>
</div>
""")

if metrics:
    best = metrics["best_model"]
    cv10 = next(r for r in metrics["cv_results"]
                if r["model"] == best and r["strategy"] == "10-fold CV" and r["metric"] == "recall")
    best_recall, best_sub = f"{cv10['mean']:.1%}", f"± {cv10['std']:.1%} · 10-fold CV"
else:
    best, best_recall, best_sub = "—", "—", "run notebook 07"

ui.kpi_row([
    ("Patients", f"{len(df):,}", "5 hospitals · USA & Europe"),
    ("Features", f"{len(C.FEATURES)}", f"{len(C.CONTINUOUS)} continuous · {len(C.CATEGORICAL)} categorical"),
    ("Best model", best, "chosen by recall + F1"),
    ("Best recall", best_recall, best_sub),
])

ui.section("How it works", "Three steps from raw data to a risk estimate.")
steps = [
    ("1", "Understand the data",
     "Descriptive statistics, skewness, correlations and outliers reveal what the 11 clinical features look like."),
    ("2", "Test & validate",
     "Hypothesis tests show which features really differ between patients. Seven models are compared with "
     "hold-out, K-fold, LOOCV and bootstrapping."),
    ("3", "Estimate risk",
     "The selected pipeline turns a patient's values into a probability, with the factors that pushed it up or down."),
]
cols = st.columns(3)
for col, (n, title, body) in zip(cols, steps):
    with col:
        ui.html(f'<div class="ci-card step"><div class="ci-step"><div class="num">{n}</div>'
                f'<div><h4>{title}</h4><p>{body}</p></div></div></div>')

ui.section("Explore", "Jump straight to a part of the project.")
links = [
    ("views/data_explorer.py", "Data Explorer", ":material/table_view:", "Filter patients, descriptive statistics, distributions"),
    ("views/statistical_tests.py", "Statistical Tests", ":material/functions:", "t-test, ANOVA, chi-square, confidence intervals"),
    ("views/model_performance.py", "Model Performance", ":material/insights:", "7 models × 4 validation strategies"),
    ("views/risk_predictor.py", "Risk Predictor", ":material/favorite:", "Estimate risk for a patient profile"),
]
cols = st.columns(4)
for col, (page, label, icon, help_text) in zip(cols, links):
    with col:
        with st.container(border=True, key=f"card-link-{label.replace(' ', '-')}"):
            st.page_link(page, label=label, icon=icon)
            st.caption(help_text)

st.write("")
ui.disclaimer()
ui.footer()
