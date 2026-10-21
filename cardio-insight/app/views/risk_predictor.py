import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import config as C
from src import plots
from src.models import local_contributions, reference_profile, risk_band

dark = ui.is_dark()
pal = plots.palette(dark)

ui.page_header("Risk Predictor", "Enter a patient's clinical values to estimate the probability of heart disease with the "
               "saved model pipeline.", "Model · Prediction")
ui.disclaimer()
st.write("")

pipe = ui.load_model()
metrics = ui.load_metrics()
if pipe is None:
    ui.missing_file("Saved model", "07_model_building_and_validation.ipynb")
    st.stop()

ref = (metrics or {}).get("reference_profile") or reference_profile(ui.load_raw())
L = C.CATEGORY_LABELS

with st.form("patient"):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Patient & symptoms")
        age = st.slider("Age (years)", 20, 90, 54)
        sex = st.selectbox("Sex", ["M", "F"], format_func=L["Sex"].get)
        cp = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"], format_func=L["ChestPainType"].get,
                          help="Asymptomatic = no chest pain. Typical angina = classic exertional chest pain.")
        angina = st.selectbox("Exercise-induced angina", ["N", "Y"], format_func=L["ExerciseAngina"].get)
        st.markdown("##### Blood tests")
        chol_known = st.checkbox("Cholesterol was measured", value=True)
        chol = st.slider("Serum cholesterol (mg/dl)", 85, 600, 237, disabled=not chol_known,
                         help="If not measured, the model fills in the training median (same as in the notebooks).")
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=L["FastingBS"].get)
    with right:
        st.markdown("##### Resting measurements")
        bp = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
        ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], format_func=L["RestingECG"].get)
        st.markdown("##### Exercise test")
        maxhr = st.slider("Maximum heart rate achieved (bpm)", 60, 205, 138)
        oldpeak = st.slider("Oldpeak – ST depression (mm)", -2.6, 6.2, 0.6, 0.1,
                            help="ST-segment depression induced by exercise relative to rest.")
        slope = st.selectbox("ST segment slope", ["Up", "Flat", "Down"], format_func=L["ST_Slope"].get)
    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

if submitted:
    patient = pd.DataFrame([{
        "Age": age, "RestingBP": bp, "Cholesterol": chol if chol_known else 0, "MaxHR": maxhr, "Oldpeak": oldpeak,
        "Sex": sex, "ChestPainType": cp, "FastingBS": fbs, "RestingECG": ecg, "ExerciseAngina": angina, "ST_Slope": slope,
    }])[C.FEATURES]
    prob, contrib = local_contributions(pipe, patient, ref)
    band = risk_band(prob)
    st.session_state["last_prediction"] = (prob, band, contrib)

if "last_prediction" in st.session_state:
    prob, band, contrib = st.session_state["last_prediction"]
    ui.section("Result")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True, key="card-gauge"):
            ui.plotly(plots.gauge(prob, dark))
            kind = {"Low": "good", "Moderate": "neutral", "High": "bad"}[band]
            ui.html(f"<div style='text-align:center'>{ui.badge(f'{band} risk', kind)}</div>")
            msg = {
                "Low": "The model sees a low likelihood of heart disease for this profile.",
                "Moderate": "The model sees a moderate likelihood. Further evaluation by a doctor would be sensible.",
                "High": "The model sees a high likelihood of heart disease. This profile should be checked by a cardiologist.",
            }[band]
            ui.html(f"<p style='text-align:center;color:var(--ci-muted);margin-top:10px'>{msg}<br>"
                    f"<span style='font-size:.8rem'>Bands: Low &lt; 30% · Moderate 30–60% · High ≥ 60%</span></p>")
    with right:
        with st.container(border=True, key="card-contrib"):
            top = contrib[contrib["contribution"].abs() > 0.0005].head(6).iloc[::-1]

            def label(row):
                v, f = row["value"], row["feature"]
                if f == "Cholesterol" and v == 0:
                    v = "not measured"
                elif f in L:
                    v = L[f].get(v, v)
                return f"{f} = {v}"

            if top.empty:
                st.markdown("**Top contributing features**")
                st.caption("Every value equals the typical patient's value, so no single feature moves the risk.")
            else:
                labels = [f"{label(r)}  ({r['contribution'] * 100:+.1f})" for _, r in top.iterrows()]
                colors = [pal[1] if c > 0 else pal[0] for c in top["contribution"]]
                fig = go.Figure(go.Bar(x=top["contribution"] * 100, y=labels, orientation="h", marker_color=colors,
                                       width=0.6, hovertemplate="%{y}<br>%{x:+.1f} percentage points<extra></extra>"))
                fig.add_vline(x=0, line=dict(color="#9AA4B2", width=1))
                span = max(5, float((top["contribution"].abs() * 100).max()) * 1.1)
                fig.update_xaxes(range=[-span, span], title_text="Change in risk (percentage points)")
                fig.update_layout(title="Top contributing features")
                ui.plotly(plots.style_fig(fig, dark, height=340, legend=False))
                st.caption("Each bar = how much the risk changes compared with replacing that one value by a typical "
                           "patient's value (median / most common). Coral raises the risk, teal lowers it.")
else:
    st.info("Fill in the form and press **Predict** to see the estimated risk.")

if metrics:
    st.caption(f"Model: {metrics['best_model']} pipeline · test recall {metrics['best']['holdout_metrics']['recall']:.1%} · "
               f"trained on {metrics['train_size']} patients. Inputs are processed locally and never stored.")
ui.footer()
