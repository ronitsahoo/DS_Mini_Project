import plotly.graph_objects as go
import streamlit as st

import ui
from src import config as C
from src import outliers as O
from src import plots

ui.require_data()
dark = ui.is_dark()
pal = plots.palette(dark)

ui.page_header("Outliers", "Compare two statistical rules with two machine-learning detectors. Cholesterol = 0 and "
               "RestingBP = 0 are missing values, not outliers, so those rows are left out here.",
               "Module 4 · Outlier detection")


@st.cache_data(show_spinner="Running the detectors…")
def run_detectors(contamination):
    df = ui.load_clean().dropna(subset=C.ZERO_AS_MISSING)
    flags = O.detect_all(df, C.ZERO_AS_MISSING, C.CONTINUOUS, contamination=contamination)
    return df, flags


with st.container(border=True, key="card-outlier-controls"):
    method = st.segmented_control("Method", O.METHODS, default="IQR") or "IQR"
    c2, c3, c4 = st.columns(3)
    x = c2.selectbox("X axis", C.CONTINUOUS, index=2)
    y = c3.selectbox("Y axis", C.CONTINUOUS, index=1)
    contamination = c4.slider("Contamination (IF / LOF)", 0.01, 0.15, 0.05, 0.01,
                              help="Expected share of outliers, used only by Isolation Forest and LOF.")

df, flags = run_detectors(contamination)

explain = {
    "IQR": "Flags a value outside **Q1 − 1.5·IQR** or **Q3 + 1.5·IQR** in Cholesterol or RestingBP. Robust, no normality assumption.",
    "Z-score": "Flags a value more than **3 standard deviations** from the mean in Cholesterol or RestingBP. "
               "Assumes a roughly normal shape; big outliers inflate the std and can hide others (masking).",
    "Isolation Forest": "Uses **all five continuous features** (standardised). Random trees isolate unusual patients in "
                        "very few splits.",
    "LOF": "Uses **all five continuous features** (standardised). Compares each patient's local density with that of "
           "their 20 nearest neighbours; much sparser = outlier.",
}
st.markdown(explain[method])

f = flags[method]
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.loc[~f, x], y=df.loc[~f, y], mode="markers", name="Normal",
                         marker=dict(size=7, color=pal[0], opacity=0.45),
                         hovertemplate=f"{x} %{{x}}<br>{y} %{{y}}<extra>normal</extra>"))
fig.add_trace(go.Scatter(x=df.loc[f, x], y=df.loc[f, y], mode="markers", name="Flagged outlier",
                         marker=dict(size=11, color=pal[1], symbol="diamond",
                                     line=dict(width=1.5, color="#152030" if dark else "white")),
                         hovertemplate=f"{x} %{{x}}<br>{y} %{{y}}<extra>outlier</extra>"))
if method == "IQR" and x in C.ZERO_AS_MISSING:
    lo, hi = O.iqr_bounds(df[x])
    for v in (lo, hi):
        fig.add_vline(x=v, line=dict(color="#9AA4B2", dash="dot", width=1.5))
if method == "IQR" and y in C.ZERO_AS_MISSING:
    lo, hi = O.iqr_bounds(df[y])
    for v in (lo, hi):
        fig.add_hline(y=v, line=dict(color="#9AA4B2", dash="dot", width=1.5))
fig.update_layout(title=f"{method}: {int(f.sum())} of {len(df)} patients flagged")
fig.update_xaxes(title_text=C.FEATURE_LABELS[x])
fig.update_yaxes(title_text=C.FEATURE_LABELS[y])
with st.container(border=True, key="card-outlier-scatter"):
    ui.plotly(plots.style_fig(fig, dark, height=440))
    if method == "IQR" and (x in C.ZERO_AS_MISSING or y in C.ZERO_AS_MISSING):
        st.caption("Dotted lines = IQR fences.")

# ---------------------------------------------------------------- comparison
ui.section("Comparison of all four methods")
left, right = st.columns([1, 1.25])
with left:
    comp = O.comparison_table(flags)
    agree = O.agreement_summary(flags)
    st.dataframe(comp.style.format({"% of rows": "{:.1f}%"}))
    st.markdown("**How many methods agree on each patient?**")
    st.dataframe(agree.T)
    st.caption(f"{int(flags.all(axis=1).sum())} patients are flagged by all four methods; "
               f"{int((~flags.any(axis=1)).sum())} by none.")
with right:
    ov = O.overlap_matrix(flags)
    scale = [[0, "#152030" if dark else "#F2F4F7"], [1, pal[0]]]
    fig = go.Figure(go.Heatmap(z=ov.values, x=ov.columns, y=ov.index, colorscale=scale, showscale=False,
                               text=ov.values, texttemplate="%{text}", textfont=dict(size=14), xgap=3, ygap=3,
                               hovertemplate="%{y} & %{x}: %{z} rows<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(title="Rows flagged by both methods")
    with st.container(border=True, key="card-overlap"):
        ui.plotly(plots.style_fig(fig, dark, height=360, legend=False))

with st.expander("Patients flagged by all four methods"):
    st.dataframe(df[flags.all(axis=1)][C.CONTINUOUS + [C.TARGET]])

ui.section("Why not just delete outliers?")
c1, c2 = st.columns(2)
with c1:
    ui.card("Statistical vs ML methods",
            "<ul><li>IQR and z-score check one feature at a time; simple and explainable.</li>"
            "<li>Isolation Forest and LOF use all features together and find unusual <i>combinations</i>, "
            "but need scaling and a contamination setting.</li></ul>")
with c2:
    ui.card("Care in medical data",
            "<ul><li>An extreme cholesterol of 500+ mg/dl can be a real, very high-risk patient.</li>"
            "<li>Only impossible values (like 0 mm Hg) are errors. The model keeps genuine outliers and uses "
            "robust preprocessing instead.</li></ul>")

ui.footer()
