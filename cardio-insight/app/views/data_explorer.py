import plotly.express as px
import streamlit as st

import ui
from src import config as C
from src import data, plots
from src import stats as S

ui.require_data()
dark = ui.is_dark()
raw = ui.load_raw()

ui.page_header("Data Explorer", "Filter the 918 patients, read the descriptive statistics computed with the "
               "hand-written formulas in <code>src/stats.py</code>, and look at any feature's distribution.",
               "Module 2.1 · Descriptive statistics")

# ---------------------------------------------------------------- filters
with st.container(border=True, key="card-filters"):
    c1, c2, c3 = st.columns([1, 1.1, 1.3])
    sex = c1.pills("Sex", ["M", "F"], selection_mode="multi", default=["M", "F"],
                   format_func=C.CATEGORY_LABELS["Sex"].get) or []
    target = c2.segmented_control("Heart disease", ["All", "Yes", "No"], default="All") or "All"
    age = c3.slider("Age range", int(raw.Age.min()), int(raw.Age.max()), (int(raw.Age.min()), int(raw.Age.max())))
    cp_opts = ["ASY", "NAP", "ATA", "TA"]
    cp = st.pills("Chest pain type", cp_opts, selection_mode="multi", default=cp_opts,
                  format_func=C.CATEGORY_LABELS["ChestPainType"].get) or []
    zeros_missing = st.toggle("Treat Cholesterol = 0 and RestingBP = 0 as missing (recommended)", value=True)

base = data.zeros_to_nan(raw) if zeros_missing else raw.copy()
mask = base.Sex.isin(sex) & base.ChestPainType.isin(cp) & base.Age.between(*age)
if target != "All":
    mask &= base[C.TARGET] == (1 if target == "Yes" else 0)
filtered = base[mask]

if len(filtered) < 10:
    st.info(f"Only {len(filtered)} patients match these filters - too few for meaningful statistics. Widen the filters.")
    st.stop()

ui.kpi_row([
    ("Patients shown", f"{len(filtered):,}", f"of {len(raw):,}"),
    ("With heart disease", f"{filtered[C.TARGET].mean():.0%}", f"{int(filtered[C.TARGET].sum())} patients"),
    ("Mean age", f"{filtered.Age.mean():.1f}", "years"),
    ("Male share", f"{(filtered.Sex == 'M').mean():.0%}", "of shown patients"),
])

# ---------------------------------------------------------------- preview
ui.section("Dataset preview", "Category codes are replaced with readable names. Missing values show as empty cells.")
st.dataframe(data.readable(filtered), hide_index=True, height=300,
             column_config={"HeartDisease": st.column_config.NumberColumn("HeartDisease", help="1 = heart disease")})

# ---------------------------------------------------------------- descriptive stats
ui.section("Descriptive statistics",
           "Computed with the manual formulas (cross-checked against pandas/scipy in notebook 02). "
           "Weighted mean = class-balanced mean (each class gets equal total weight).")
if filtered[C.TARGET].nunique() == 2:
    weights = S.class_balanced_weights(filtered[C.TARGET])
else:
    weights = None
table = S.descriptive_table(filtered, C.CONTINUOUS, weights)
show = table.drop(columns=["Min", "Max"]).T
show = show.map(lambda v: f"{v:,.3f}" if isinstance(v, float) else str(v)).astype(str)
st.dataframe(show, height=(len(show) + 1) * 35 + 3,
             column_config={"_index": st.column_config.TextColumn("Measure", width="medium")})

cols = st.columns(len(C.CONTINUOUS))
for col, feat in zip(cols, C.CONTINUOUS):
    sk = table.loc[feat, "Pearson skew"]
    col.markdown(f"**{feat}**  \n<span style='color:var(--ci-muted);font-size:.85rem'>{S.skew_label(sk).capitalize()} · "
                 f"{table.loc[feat, 'Kurtosis type'].lower()}</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------- distribution
ui.section("Distribution of a feature", "Split by heart disease status.")
c1, c2 = st.columns([1, 1])
feat = c1.selectbox("Feature", C.CONTINUOUS, index=3, format_func=lambda f: C.FEATURE_LABELS[f])
kind = c2.segmented_control("Chart", ["Histogram", "Box plot"], default="Histogram")

plot_df = filtered.dropna(subset=[feat]).copy()
plot_df["Status"] = plot_df[C.TARGET].map(C.CLASS_NAMES)
colors = plots.class_colors(dark)
order = {"Status": [C.CLASS_NAMES[0], C.CLASS_NAMES[1]]}
if kind == "Box plot":
    fig = px.box(plot_df, x="Status", y=feat, color="Status", color_discrete_map=colors, points="outliers",
                 category_orders=order, labels={feat: C.FEATURE_LABELS[feat], "Status": ""})
    fig.update_traces(marker=dict(size=5), line=dict(width=2))
else:
    fig = px.histogram(plot_df, x=feat, color="Status", color_discrete_map=colors, barmode="overlay",
                       nbins=35, opacity=0.6, marginal="box", category_orders=order,
                       labels={feat: C.FEATURE_LABELS[feat]})
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="Patients", row=1, col=1)
with st.container(border=True, key="card-dist"):
    ui.plotly(plots.style_fig(fig, dark, height=420))
    m = plot_df.groupby(C.TARGET)[feat].agg(["mean", "median"])
    if len(m) == 2:
        st.caption(f"Mean {feat}: {m.loc[0, 'mean']:.1f} (no heart disease) vs {m.loc[1, 'mean']:.1f} (heart disease) · "
                   f"medians {m.loc[0, 'median']:.1f} vs {m.loc[1, 'median']:.1f}")

ui.footer()
