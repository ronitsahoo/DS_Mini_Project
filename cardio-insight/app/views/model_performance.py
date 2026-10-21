import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import plots
from src.models import METRIC_NAMES

dark = ui.is_dark()
pal = plots.palette(dark)
metrics = ui.load_metrics()

ui.page_header("Model Performance", "Seven classifiers, each wrapped in a leakage-free sklearn Pipeline, compared with "
               "four validation strategies. Results are read from <code>models/metrics.json</code> - nothing is retrained here.",
               "Module 3.1 · Model validation")

if metrics is None:
    ui.missing_file("Model metrics", "07_model_building_and_validation.ipynb")
    st.stop()

best = metrics["best_model"]
bm = metrics["best"]["holdout_metrics"]
ci = metrics["best"]["recall_ci"]
res = pd.DataFrame(metrics["cv_results"])
strategies = metrics["strategies"]

ui.kpi_row([
    ("Selected model", best, "highest (recall + F1) / 2, 10-fold CV"),
    ("Test recall", f"{bm['recall']:.1%}", f"95% CI {ci['lower']:.1%} – {ci['upper']:.1%}"),
    ("Test F1", f"{bm['f1']:.3f}", f"precision {bm['precision']:.3f}"),
    ("Test ROC-AUC", f"{bm['roc_auc']:.3f}", f"{metrics['test_size']} unseen patients"),
])

# ---------------------------------------------------------------- comparison table
ui.section("Models × validation strategies", "LOOCV was run only for Logistic Regression and Naive Bayes (918 fits each).")
metric = st.segmented_control("Metric", list(METRIC_NAMES), default="recall", format_func=METRIC_NAMES.get) or "recall"
sub = res[res.metric == metric]
mean_t = sub.pivot(index="model", columns="strategy", values="mean")[strategies]
std_t = sub.pivot(index="model", columns="strategy", values="std")[strategies]
order = mean_t["10-fold CV"].sort_values(ascending=False).index
display = mean_t.loc[order].copy().astype(object)
for m in order:
    for s in strategies:
        v, sd = mean_t.loc[m, s], std_t.loc[m, s]
        display.loc[m, s] = "—" if pd.isna(v) else (f"{v:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{v:.3f}")
display.index = [f"★ {m}" if m == best else m for m in display.index]
st.dataframe(display, height=300)

# ---------------------------------------------------------------- grouped bar chart
fig = go.Figure()
for i, s in enumerate(strategies):
    vals = mean_t.loc[order, s]
    fig.add_trace(go.Bar(x=list(order), y=vals, name=s, marker_color=pal[i],
                         error_y=dict(type="data", array=std_t.loc[order, s].fillna(0), color="#9AA4B2",
                                      thickness=1, width=3, visible=s in ("5-fold CV", "10-fold CV")),
                         hovertemplate="%{x}<br>" + s + ": %{y:.3f}<extra></extra>"))
lo = max(0.0, float(mean_t.min().min()) - 0.08)
fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06, title=f"{METRIC_NAMES[metric]} by model and strategy")
fig.update_yaxes(range=[lo, 1.0], title_text=METRIC_NAMES[metric])
with st.container(border=True, key="card-bars"):
    ui.plotly(plots.style_fig(fig, dark, height=420))
    st.caption("Error bars = ± 1 standard deviation across folds (K-fold only). Hold-out and LOOCV give a single number.")

# ---------------------------------------------------------------- confusion matrix + ROC
ui.section(f"{best} on the hold-out test set")
left, right = st.columns(2)
with left:
    cm = metrics["best"]["confusion_matrix"]
    labels = [["True negative", "False positive<br>Type I"], ["False negative<br>Type II", "True positive"]]
    text = [[f"<b style='font-size:22px'>{cm[r][c]}</b><br>{labels[r][c]}" for c in range(2)] for r in range(2)]
    base = "#1A2638" if dark else "#EEF1F4"
    fig = go.Figure(go.Heatmap(z=[[1, 0], [0, 1]], x=["No HD", "HD"], y=["No HD", "HD"],
                               colorscale=[[0, base], [1, pal[0]]], showscale=False,
                               text=text, texttemplate="%{text}", xgap=4, ygap=4, hoverinfo="skip",
                               textfont=dict(size=12, color="#FFFFFF") if dark else dict(size=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False, title_text="Actual")
    fig.update_xaxes(showgrid=False, side="top", title_text="Predicted")
    with st.container(border=True, key="card-cm"):
        ui.plotly(plots.style_fig(fig, dark, height=360, legend=False))
        st.caption(f"Recall {bm['recall']:.3f} · specificity {bm['specificity']:.3f} · precision {bm['precision']:.3f}")
with right:
    with st.container(border=True, key="card-roc"):
        show_all = st.toggle("Show other models", value=True)
        fig = go.Figure()
        if show_all:
            for name, roc in metrics["roc_all"].items():
                if name != best:
                    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=name,
                                             line=dict(color="#4A5568" if dark else "#CBD2DB", width=1.2),
                                             showlegend=False,
                                             hovertemplate=f"{name} (AUC {roc['auc']:.3f})<extra></extra>"))
        roc = metrics["roc_all"][best]
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{best} (AUC {roc['auc']:.3f})",
                                 line=dict(color=pal[0], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(color="#9AA4B2", dash="dash", width=1.2)))
        fig.update_xaxes(title_text="False positive rate", range=[0, 1])
        fig.update_yaxes(title_text="True positive rate (recall)", range=[0, 1.02])
        ui.plotly(plots.style_fig(fig, dark, height=330))

# ---------------------------------------------------------------- Type I/II explainer
ui.section("Type I vs Type II errors", "Why this project chooses models by recall first.")
c1, c2, c3 = st.columns(3)
with c1:
    ui.card(f"Type I · false positive ({cm[0][1]})",
            "<p>A <b>healthy</b> person is told they may have heart disease. Result: anxiety and extra tests "
            "(ECG, stress test, angiography). Unpleasant and costly - but not dangerous.</p>")
with c2:
    ui.card(f"Type II · false negative ({cm[1][0]})",
            "<p>A patient who <b>has</b> heart disease is sent home as healthy. Treatment is delayed and a heart attack "
            "may follow. <b>This is the dangerous error.</b></p>")
with c3:
    ui.card("So we maximise recall",
            f"<p>Recall = TP / (TP + FN) = share of sick patients caught. The {best} catches "
            f"<b>{bm['recall']:.1%}</b> on the test set (95% bootstrap CI {ci['lower']:.1%}–{ci['upper']:.1%}). "
            "F1 is used alongside so false alarms stay reasonable.</p>")

# ---------------------------------------------------------------- importance
ui.section("What drives the model?", "Permutation importance: the drop in ROC-AUC when one feature is shuffled.")
imp = pd.DataFrame(metrics["best"]["permutation_importance"]).sort_values("importance")
fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=pal[0],
                       error_x=dict(type="data", array=imp["std"], color="#9AA4B2", thickness=1),
                       hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
fig.update_xaxes(title_text="Drop in ROC-AUC")
with st.container(border=True, key="card-imp"):
    ui.plotly(plots.style_fig(fig, dark, height=380, legend=False))

with st.expander("Selection ranking (10-fold CV)"):
    rank = pd.DataFrame(metrics["ranking"]).set_index("model")
    st.dataframe(rank.style.format("{:.4f}"))
    st.caption(metrics["selection_rule"])

ui.footer()
