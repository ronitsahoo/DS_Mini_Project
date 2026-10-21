"""Plotting helpers.

* Matplotlib/Seaborn functions are used in the notebooks (and saved as PNGs to reports/figures).
* Plotly functions are used in the Streamlit app.
Both use the same validated palette from config.py so the project looks consistent.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats as sps

from .config import (CLASS_COLORS, CLASS_NAMES, FIGURES_DIR, NAVY, PALETTE_DARK, PALETTE_LIGHT,
                     TARGET, TEAL)

# Diverging scale for correlations: coral (negative) -> neutral grey -> teal (positive)
DIVERGING = [[0.0, "#C2552E"], [0.5, "#EEF0F2"], [1.0, "#007A70"]]
DIVERGING_DARK = [[0.0, "#D96F45"], [0.5, "#2A3445"], [1.0, "#1FA398"]]


# =====================================================================
# Matplotlib / Seaborn (notebooks)
# =====================================================================
def set_style():
    """Apply one clean style to every matplotlib figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": "#C9CED6", "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
        "axes.titleweight": "bold", "xtick.color": "#4A5568", "ytick.color": "#4A5568",
        "grid.color": "#E6E9EE", "axes.prop_cycle": plt.cycler(color=PALETTE_LIGHT),
        "font.family": "sans-serif",
    })


def save_fig(fig, name):
    """Save a figure to reports/figures/<name>.png."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def hist_kde_grid(df, columns, hue=TARGET):
    """Histogram + KDE for each column, split by the target."""
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.6))
    for ax, col in zip(np.atleast_1d(axes), columns):
        sns.histplot(data=df, x=col, hue=hue, kde=True, ax=ax, palette=CLASS_COLORS,
                     element="step", stat="density", common_norm=False, alpha=0.25)
        ax.set_title(col)
        ax.set_ylabel("Density")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("")
            for t, lab in zip(leg.texts, CLASS_NAMES.values()):
                t.set_text(lab)
    fig.tight_layout()
    return fig


def box_grid(df, columns, hue=TARGET):
    """Box plot for each column, split by the target."""
    fig, axes = plt.subplots(1, len(columns), figsize=(3.4 * len(columns), 3.8))
    for ax, col in zip(np.atleast_1d(axes), columns):
        sns.boxplot(data=df, x=hue, y=col, hue=hue, ax=ax, palette=CLASS_COLORS, legend=False,
                    width=0.5, fliersize=3)
        ax.set_xticks([0, 1], ["No HD", "HD"])
        ax.set_xlabel("")
        ax.set_title(col)
    fig.tight_layout()
    return fig


def qq_grid(df, columns):
    """Q-Q plot of each column against a normal distribution."""
    fig, axes = plt.subplots(1, len(columns), figsize=(3.6 * len(columns), 3.6))
    for ax, col in zip(np.atleast_1d(axes), columns):
        x = df[col].dropna()
        (osm, osr), (slope, intercept, _) = sps.probplot(x, dist="norm")
        ax.scatter(osm, osr, s=8, color=TEAL, alpha=0.6)
        ax.plot(osm, slope * np.asarray(osm) + intercept, color=PALETTE_LIGHT[1], lw=2)
        ax.set_title(col)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
    fig.tight_layout()
    return fig


def corr_heatmap(matrix, title="Pearson correlation", fmt=".2f", center=0, cmap=None):
    fig, ax = plt.subplots(figsize=(7, 5.8))
    cmap = cmap or sns.diverging_palette(20, 180, s=75, l=50, as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, center=center, square=True,
                linewidths=2, linecolor="white", cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# =====================================================================
# Plotly (Streamlit app)
# =====================================================================
def palette(dark=False):
    return PALETTE_DARK if dark else PALETTE_LIGHT


def class_colors(dark=False):
    p = palette(dark)
    return {CLASS_NAMES[0]: p[0], CLASS_NAMES[1]: p[1]}


def style_fig(fig, dark=False, height=380, legend=True):
    """Shared Plotly layout: transparent background, recessive grid, Inter font."""
    ink = "#E6EAF0" if dark else NAVY
    muted = "#9AA4B2" if dark else "#5B6676"
    grid = "rgba(255,255,255,0.08)" if dark else "#E9ECF0"
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=(78 if has_title and legend else 56 if has_title or legend else 16), b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=ink, size=13),
        title=dict(x=0, xanchor="left", y=1, yanchor="top", yref="container", pad=dict(t=10, l=4),
                   font=dict(size=15, color=ink)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text="",
                    font=dict(color=muted)),
        showlegend=legend,
        hoverlabel=dict(font_family="Inter, 'Segoe UI', sans-serif"),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=dict(color=muted), automargin=True,
                     title_font=dict(color=muted, size=12))
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=dict(color=muted), automargin=True,
                     title_font=dict(color=muted, size=12))
    return fig


def gauge(prob, dark=False):
    """Probability gauge with Low / Moderate / High bands."""
    ink = "#E6EAF0" if dark else NAVY
    band_alpha = "33" if dark else "26"
    p = palette(dark)
    colour = p[0] if prob < 0.30 else p[3] if prob < 0.60 else p[1]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": ink}, "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9AA4B2",
                     "tickvals": [0, 30, 60, 100]},
            "bar": {"color": colour, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": p[0] + band_alpha},
                {"range": [30, 60], "color": p[3] + band_alpha},
                {"range": [60, 100], "color": p[1] + band_alpha},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=36, r=36, t=24, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=ink))
    return fig
