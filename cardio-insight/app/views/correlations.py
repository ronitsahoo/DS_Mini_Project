import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from src import config as C
from src import plots
from src import stats as S

ui.require_data()
dark = ui.is_dark()
df = ui.load_clean()
cols = C.CONTINUOUS + [C.TARGET]

ui.page_header("Correlations", "How strongly do the continuous features move together - and with heart disease? "
               "Karl Pearson's r ranges from −1 (perfect negative) through 0 (none) to +1 (perfect positive).",
               "Module 2.1 · Correlation analysis")

c1, c2 = st.columns([1, 1])
matrix_kind = c1.segmented_control("Matrix", ["Pearson correlation", "Covariance"], default="Pearson correlation") \
    or "Pearson correlation"

# ---------------------------------------------------------------- heatmap
if matrix_kind == "Pearson correlation":
    m = df[cols].corr()
    z, zmin, zmax, fmt = m.values, -1, 1, ".2f"
    scale = plots.DIVERGING_DARK if dark else plots.DIVERGING
    caption = "Teal = positive, coral = negative, neutral = close to zero. Missing values skipped pair-wise."
else:
    m = df[cols].cov()
    # colour by the correlation so the scale is comparable, but print the covariance values
    z, zmin, zmax, fmt = df[cols].corr().values, -1, 1, ".1f"
    scale = plots.DIVERGING_DARK if dark else plots.DIVERGING
    caption = ("Numbers are covariances (they depend on units - Cholesterol's are huge, Oldpeak's tiny). "
               "Colours show the matching correlation, which is the unit-free version.")

text = np.vectorize(lambda v: format(v, fmt))(m.values)
fig = go.Figure(go.Heatmap(z=z, x=m.columns, y=m.index, zmin=zmin, zmax=zmax, colorscale=scale,
                           text=text, texttemplate="%{text}", textfont=dict(size=13),
                           xgap=3, ygap=3, colorbar=dict(thickness=10, len=0.8, outlinewidth=0),
                           hovertemplate="%{y} × %{x}<br>value %{text}<extra></extra>"))
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=False, side="bottom")
with st.container(border=True, key="card-heatmap"):
    ui.plotly(plots.style_fig(fig, dark, height=470, legend=False))
    st.caption(caption)

# ---------------------------------------------------------------- strongest pairs
corr = df[cols].corr()
pairs = []
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        pairs.append((a, b, corr.loc[a, b]))
pairs.sort(key=lambda t: t[2])
neg, pos = pairs[0], pairs[-1]
none_pair = min(pairs, key=lambda t: abs(t[2]))
k1, k2, k3 = st.columns(3)
for col, (title, (a, b, r)) in zip([k1, k2, k3], [("Strongest positive", pos), ("Strongest negative", neg),
                                                   ("Closest to zero", none_pair)]):
    with col:
        ui.html(ui.kpi(title, f"{r:+.2f}", f"{a} × {b} · {S.correlation_strength(r)}"))

# ---------------------------------------------------------------- scatter matrix
ui.section("Scatter matrix", "Every pair of selected features, coloured by heart disease status.")
chosen = st.multiselect("Features", C.CONTINUOUS, default=["Age", "MaxHR", "Oldpeak", "Cholesterol"],
                        max_selections=5)
if len(chosen) >= 2:
    sdf = df.dropna(subset=chosen).copy()
    sdf["Status"] = sdf[C.TARGET].map(C.CLASS_NAMES)
    fig = px.scatter_matrix(sdf, dimensions=chosen, color="Status", color_discrete_map=plots.class_colors(dark),
                            category_orders={"Status": list(C.CLASS_NAMES.values())}, opacity=0.55)
    fig.update_traces(diagonal_visible=False, showupperhalf=False,
                      marker=dict(size=4.5, line=dict(width=0.4, color="#0E1520" if dark else "white")))
    with st.container(border=True, key="card-splom"):
        ui.plotly(plots.style_fig(fig, dark, height=170 * len(chosen) + 60))
else:
    st.info("Pick at least two features.")

# ---------------------------------------------------------------- two-variable view
ui.section("Look at one pair", "With a least-squares trend line.")
c1, c2 = st.columns(2)
x = c1.selectbox("X axis", C.CONTINUOUS, index=0)
y = c2.selectbox("Y axis", C.CONTINUOUS, index=3)
if x == y:
    st.info("Choose two different features.")
else:
    pdf = df.dropna(subset=[x, y]).copy()
    pdf["Status"] = pdf[C.TARGET].map(C.CLASS_NAMES)
    r = S.pearson_r(pdf[x], pdf[y])
    slope, intercept = np.polyfit(pdf[x], pdf[y], 1)
    xs = np.linspace(pdf[x].min(), pdf[x].max(), 50)
    fig = px.scatter(pdf, x=x, y=y, color="Status", color_discrete_map=plots.class_colors(dark), opacity=0.6,
                     category_orders={"Status": list(C.CLASS_NAMES.values())},
                     labels={x: C.FEATURE_LABELS[x], y: C.FEATURE_LABELS[y]})
    fig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="#0E1520" if dark else "white")))
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend line",
                             line=dict(color="#9AA4B2", width=2, dash="dash")))
    with st.container(border=True, key="card-pair"):
        ui.plotly(plots.style_fig(fig, dark, height=400))
        st.markdown(f"**r = {r:+.3f}** → {S.correlation_strength(r)}. "
                    "Remember: correlation shows that two variables move together, **not** that one causes the other.")

ui.footer()
