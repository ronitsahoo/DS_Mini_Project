import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import plots

dark = ui.is_dark()
pal = plots.palette(dark)
df = ui.load_raw()
metrics = ui.load_metrics()

ui.page_header("Ethics & About", "A model can be accurate and still be unfair or unsafe. This page summarises "
               "notebook 08 and credits the work this project builds on.", "Module 6.2 · Ethics in data science")

# ---------------------------------------------------------------- bias
ui.section("Dataset bias")
n_m, n_f = int((df.Sex == "M").sum()), int((df.Sex == "F").sum())
fair = (metrics or {}).get("fairness")
cols = st.columns(4)
items = [
    ("Male patients", f"{n_m}", f"{n_m / len(df):.0%} of the dataset"),
    ("Female patients", f"{n_f}", f"{n_f / len(df):.0%} of the dataset"),
]
if fair:
    g = fair["gaps"]["10-fold CV (all patients)"]
    items += [("Recall · men", f"{g['male_recall']:.1%}", "10-fold CV"),
              ("Recall · women", f"{g['female_recall']:.1%}", f"gap {g['gap'] * 100:+.1f} points")]
for col, it in zip(cols, items):
    with col:
        ui.html(ui.kpi(*it))

left, right = st.columns(2, gap="large")
with left:
    st.write("")
    ui.card("Why results may not transfer to Indian patients",
            "<ul><li>All 918 patients come from <b>US and European hospitals</b> (Cleveland, Hungary, Switzerland, "
            "Long Beach VA, Statlog), mostly collected in the 1980s.</li>"
            "<li>South Asians often develop coronary disease <b>5–10 years earlier</b> and at lower cholesterol and BMI, "
            "so learned cut-offs may not apply.</li>"
            "<li>Patients were <b>referred to cardiology clinics</b>: 55% have heart disease, far above the general "
            "population, so risks would be over-estimated in general screening.</li>"
            "<li>Only <b>21% are women</b>, and only 50 sick women exist in the data.</li></ul>")
with right:
    if fair:
        tbl = pd.DataFrame(fair["table"])
        fig = go.Figure()
        for i, sex in enumerate(["Male", "Female"]):
            s = tbl[tbl.Sex == sex]
            short = s["Evaluation"].map({"Hold-out test set": "Hold-out test", "10-fold CV (all patients)": "10-fold CV"})
            fig.add_trace(go.Bar(x=short, y=s["Recall"], name=sex, marker_color=pal[2 + i],
                                 text=[f"{v:.2f}" for v in s["Recall"]], textposition="outside", cliponaxis=False,
                                 textfont=dict(size=13)))
        fig.update_layout(barmode="group", title="Recall by sex", bargap=0.35)
        fig.update_yaxes(range=[0.5, 1.05], title_text="Recall")
        with st.container(border=True, key="card-fair"):
            ui.plotly(plots.style_fig(fig, dark, height=330))
            st.caption("The model misses more sick women than sick men. The hold-out set contains only 6 sick women, "
                       "so the 10-fold CV estimate is more reliable.")
    else:
        ui.missing_file("Fairness results", "08_ethics_and_limitations.ipynb")

# ---------------------------------------------------------------- principles
ui.section("Responsible use")
c1, c2, c3 = st.columns(3)
with c1:
    ui.card("Privacy",
            "<ul><li>Health data is highly sensitive (India: DPDP Act 2023; EU: GDPR; US: HIPAA).</li>"
            "<li>This dataset is anonymised - no names, IDs or dates.</li>"
            "<li>Patients must give informed consent.</li>"
            "<li>The app runs locally and stores nothing you enter.</li></ul>")
with c2:
    ui.card("Never a replacement for a doctor",
            "<ul><li>Uses only 11 numbers; a doctor sees the whole patient.</li>"
            "<li>Still misses about 1 in 10 sick patients.</li>"
            "<li>Never clinically validated.</li>"
            "<li>At most a decision-support aid, always with a disclaimer.</li></ul>")
with c3:
    ui.card("Reproducibility",
            "<ul><li><code>random_state=42</code> everywhere.</li>"
            "<li>Pinned versions in <code>requirements.txt</code>.</li>"
            "<li>All notebooks rerun top-to-bottom to the same numbers.</li>"
            "<li>Shared code in <code>src/</code> for notebooks and app.</li></ul>")

# ---------------------------------------------------------------- credits
ui.section("Credits & acknowledgements")
c1, c2 = st.columns(2)
with c1:
    ui.card("Dataset",
            "<p><b>Heart Failure Prediction Dataset</b> by fedesoriano (Kaggle, 2021) - "
            "<a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction' target='_blank'>kaggle.com/datasets/"
            "fedesoriano/heart-failure-prediction</a>. Combines the Cleveland, Hungarian, Switzerland, Long Beach VA and "
            "Statlog heart disease datasets from the UCI Machine Learning Repository.</p>")
with c2:
    ui.card("Original repository",
            "<p>This project started from <a href='https://github.com/nripstein/Heart-Disease-Prediction' target='_blank'>"
            "nripstein/Heart-Disease-Prediction</a> (MIT License). It was restructured and extended with manual statistics, "
            "inferential tests, outlier detection, leakage-free pipelines, multiple validation strategies, an ethics analysis "
            "and this redesigned multi-page app. The original code is kept in <code>_original_backup/</code>.</p>")

a = ui.AUTHOR
st.write("")
ui.card("About this project",
        f"<p>Data Science Mini Project (CA2) · {a['dept']}<br>"
        f"Author: <b>{a['name']}</b> · Roll no. {a['roll']} · Class/Batch {a['batch']} · Guide: {a['guide']}</p>"
        + (f"<p style='margin-top:6px'>Built with Python {metrics['versions']['python']}, scikit-learn "
           f"{metrics['versions']['scikit-learn']}, XGBoost {metrics['versions']['xgboost']}, pandas "
           f"{metrics['versions']['pandas']}, Streamlit and Plotly.</p>" if metrics else ""))

ui.footer()
