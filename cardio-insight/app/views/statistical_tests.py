import pandas as pd
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
pal = plots.palette(dark)
colors = plots.class_colors(dark)

ui.page_header("Statistical Tests", "Pick a feature and see the hypotheses, the test statistic, the p-value and what it "
               "means in plain English. Significance level α = 0.05 throughout.", "Module 2.2 · Inferential statistics")

t_tab, a_tab, c_tab, ci_tab = st.tabs(["Welch t-test", "One-way ANOVA", "Chi-square", "Confidence intervals"])

GROUPS = ["HeartDisease", "ChestPainType", "ST_Slope", "RestingECG", "Sex", "ExerciseAngina"]


def readable_groups(frame, col):
    out = frame.copy()
    mapping = {C.TARGET: C.CLASS_NAMES, **C.CATEGORY_LABELS}.get(col, {})
    out["Group"] = out[col].map(mapping).fillna(out[col].astype(str))
    return out


# ---------------------------------------------------------------- t-test
with t_tab:
    st.markdown("Compares the **mean** of a continuous feature between patients **without** and **with** heart disease. "
                "Welch's version does not assume equal variances.")
    feat = st.selectbox("Feature", C.CONTINUOUS, index=3, key="t_feat", format_func=lambda f: C.FEATURE_LABELS[f])
    r = S.welch_ttest(df, feat)
    left, right = st.columns([1.25, 1])
    with left:
        ui.test_result(r, "t statistic")
    with right:
        with st.container(border=True, key="card-t-chart"):
            pdf = readable_groups(df.dropna(subset=[feat]), C.TARGET)
            fig = px.box(pdf, x="Group", y=feat, color="Group", color_discrete_map=colors,
                         category_orders={"Group": list(C.CLASS_NAMES.values())},
                         labels={"Group": "", feat: C.FEATURE_LABELS[feat]})
            means = pdf.groupby("Group")[feat].mean()
            fig.add_trace(go.Scatter(x=means.index, y=means.values, mode="markers", name="Mean",
                                     marker=dict(symbol="diamond", size=11, color="#E6EAF0" if dark else C.NAVY,
                                                 line=dict(width=2, color=pal[0]))))
            ui.plotly(plots.style_fig(fig, dark, height=340, legend=False))
            st.caption(f"Group sizes: {r['n_0']} vs {r['n_1']} patients · diamond = group mean")

# ---------------------------------------------------------------- ANOVA
with a_tab:
    st.markdown("Tests whether the **means of 2 or more groups** are equal, by comparing variation *between* groups "
                "with variation *within* groups (F = between / within).")
    c1, c2 = st.columns(2)
    feat = c1.selectbox("Continuous feature", C.CONTINUOUS, index=3, key="a_feat", format_func=lambda f: C.FEATURE_LABELS[f])
    grp = c2.selectbox("Grouping variable", GROUPS, index=1, key="a_grp", format_func=lambda g: C.FEATURE_LABELS[g])
    r = S.anova_test(df, feat, grp)
    left, right = st.columns([1.25, 1])
    with left:
        ui.test_result(r, "F statistic")
    with right:
        with st.container(border=True, key="card-a-chart"):
            pdf = readable_groups(df.dropna(subset=[feat]), grp)
            stats_g = pdf.groupby("Group")[feat].agg(["mean", "count", "std"]).sort_values("mean")
            stats_g["ci"] = 1.96 * stats_g["std"] / stats_g["count"] ** 0.5
            fig = go.Figure(go.Bar(x=stats_g["mean"], y=stats_g.index, orientation="h", marker_color=pal[0],
                                   error_x=dict(type="data", array=stats_g["ci"], color="#9AA4B2", thickness=1.5),
                                   text=[f"{m:.1f}" for m in stats_g["mean"]], textposition="inside",
                                   insidetextanchor="end", textfont=dict(color="white"),
                                   hovertemplate="%{y}: mean %{x:.1f}<extra></extra>"))
            fig.update_layout(title=f"Mean {feat} by {grp} (±95% CI)")
            fig.update_xaxes(title_text=C.FEATURE_LABELS[feat])
            ui.plotly(plots.style_fig(fig, dark, height=340, legend=False))
            st.caption("Groups: " + ", ".join(f"{g} (n={int(n)})" for g, n in stats_g["count"].items()))

# ---------------------------------------------------------------- chi-square
with c_tab:
    st.markdown("Tests whether a **categorical** feature and heart disease are **independent**, by comparing observed "
                "counts with the counts expected under independence.")
    feat = st.selectbox("Categorical feature", C.CATEGORICAL, index=5, key="c_feat", format_func=lambda f: C.FEATURE_LABELS[f])
    r = S.chi_square_test(df, feat)
    left, right = st.columns([1.25, 1])
    with left:
        ui.test_result(r, "χ² statistic")
        st.caption(f"Degrees of freedom = {r['dof']} · Cramér's V = {r['cramers_v']:.3f} "
                   "(strength of association: 0 = none, 1 = perfect)")
    with right:
        with st.container(border=True, key="card-c-chart"):
            tbl = r["table"].copy()
            labels = C.CATEGORY_LABELS.get(feat, {})
            tbl.index = [labels.get(i, str(i)) for i in tbl.index]
            rate = (tbl[1] / tbl.sum(axis=1)).sort_values()
            fig = go.Figure()
            fig.add_trace(go.Bar(y=rate.index, x=tbl.loc[rate.index, 0], name=C.CLASS_NAMES[0], orientation="h",
                                 marker=dict(color=pal[0], line=dict(color="rgba(0,0,0,0)", width=0))))
            fig.add_trace(go.Bar(y=rate.index, x=tbl.loc[rate.index, 1], name=C.CLASS_NAMES[1], orientation="h",
                                 marker=dict(color=pal[1], line=dict(color="rgba(0,0,0,0)", width=0))))
            fig.update_layout(barmode="stack", title=f"Patients by {feat}")
            fig.update_xaxes(title_text="Patients")
            ui.plotly(plots.style_fig(fig, dark, height=320))
        obs = r["table"].rename(columns=C.CLASS_NAMES)
        exp = r["expected"].rename(columns=C.CLASS_NAMES).round(1)
        obs.index = [labels.get(i, str(i)) for i in obs.index]
        exp.index = obs.index
        st.markdown("**Observed vs expected counts**")
        st.dataframe(pd.concat({"Observed": obs, "Expected if independent": exp}, axis=1))

# ---------------------------------------------------------------- confidence intervals
with ci_tab:
    st.markdown("A confidence interval gives a range of plausible values for the **true population mean**: "
                "mean ± t × s / √n.")
    c1, c2 = st.columns(2)
    feat = c1.selectbox("Feature", C.CONTINUOUS, index=3, key="ci_feat", format_func=lambda f: C.FEATURE_LABELS[f])
    conf = c2.segmented_control("Confidence level", [0.90, 0.95, 0.99], default=0.95,
                                format_func=lambda v: f"{v:.0%}", key="ci_conf") or 0.95
    rows = []
    for g in [0, 1]:
        ci = S.mean_ci(df.loc[df[C.TARGET] == g, feat], conf)
        rows.append({"Group": C.CLASS_NAMES[g], **ci})
    cis = pd.DataFrame(rows)
    overlap = not (cis.loc[0, "upper"] < cis.loc[1, "lower"] or cis.loc[1, "upper"] < cis.loc[0, "lower"])
    left, right = st.columns([1, 1.15])
    with left:
        with st.container(border=True, key="card-ci-text"):
            for _, row in cis.iterrows():
                ui.html(f"<div style='margin-bottom:12px'><div class='ci-test-grid' style='margin:0'>"
                        f"<div><div class='k'>{row['Group']}</div><div class='ci-stat'>{row['mean']:.2f}</div></div>"
                        f"<div><div class='k'>{conf:.0%} CI</div><div class='v' style='font-size:1.1rem;font-weight:600'>"
                        f"[{row['lower']:.2f}, {row['upper']:.2f}]</div></div></div>"
                        f"<div style='color:var(--ci-muted);font-size:.85rem'>n = {row['n']}, standard error = {row['se']:.3f}, "
                        f"t critical = {row['t_crit']:.3f}</div></div>")
            verdict = (ui.badge("Intervals overlap", "bad") if overlap else ui.badge("Intervals do not overlap", "good"))
            ui.html(verdict)
            msg = ("The intervals overlap, so this view alone does not show a clear difference between the groups."
                   if overlap else
                   f"We are {conf:.0%} confident the true mean {feat} lies in each range. The ranges do not overlap, "
                   f"so the two groups really differ in average {feat}.")
            ui.html(f"<div class='ci-conclusion' style='margin-top:12px'>{msg}</div>")
    with right:
        with st.container(border=True, key="card-ci-chart"):
            fig = go.Figure()
            for i, row in cis.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["mean"]], y=[row["Group"]], mode="markers", name=row["Group"],
                    marker=dict(size=13, color=pal[i], line=dict(width=2, color="#152030" if dark else "white")),
                    error_x=dict(type="data", symmetric=False, array=[row["upper"] - row["mean"]],
                                 arrayminus=[row["mean"] - row["lower"]], color=pal[i], thickness=2.5, width=8),
                    hovertemplate=f"{row['Group']}<br>mean %{{x:.2f}}<br>CI [{row['lower']:.2f}, {row['upper']:.2f}]<extra></extra>"))
            fig.update_layout(title=f"Mean {feat} with {conf:.0%} confidence interval")
            fig.update_xaxes(title_text=C.FEATURE_LABELS[feat])
            ui.plotly(plots.style_fig(fig, dark, height=300, legend=False))

ui.footer()
