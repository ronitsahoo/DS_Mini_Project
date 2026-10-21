"""Shared UI helpers for the CardioInsight Streamlit app: styling, cards, loaders.

Every page imports this module. It also puts the project root on sys.path so
the pages can use the same `src/` code as the notebooks.
"""
import re
import sys
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from src import config as C  # noqa: E402
from src import data  # noqa: E402
from src.models import load_json  # noqa: E402

# ---------------------------------------------------------------- author details (edit these)
AUTHOR = {
    "name": "[YOUR NAME]",
    "roll": "[ROLL NO]",
    "batch": "[BATCH]",
    "guide": "[FACULTY NAME]",
    "dept": "Department of Information Technology, VESIT Mumbai",
}


# ---------------------------------------------------------------- theme
def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


_LIGHT = """
  --ci-bg:#F7F8FA; --ci-surface:#FFFFFF; --ci-surface-2:#F2F4F7; --ci-border:#E3E7EC;
  --ci-ink:#1B2A41; --ci-muted:#5B6676; --ci-faint:#8A94A3;
  --ci-accent:#008C80; --ci-accent-ink:#00756B; --ci-accent-soft:rgba(0,140,128,.09);
  --ci-good:#17734A; --ci-good-bg:#E5F4EC; --ci-bad:#B42318; --ci-bad-bg:#FDECEA;
  --ci-warn:#8A5A00; --ci-warn-bg:#FFF4DB; --ci-warn-border:#F2D58A;
  --ci-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 20px rgba(16,24,40,.05);
"""
_DARK = """
  --ci-bg:#0E1520; --ci-surface:#152030; --ci-surface-2:#1A2638; --ci-border:#263243;
  --ci-ink:#E6EAF0; --ci-muted:#9AA4B2; --ci-faint:#6F7A8A;
  --ci-accent:#1FA398; --ci-accent-ink:#4FC2B8; --ci-accent-soft:rgba(31,163,152,.14);
  --ci-good:#5CCB8F; --ci-good-bg:rgba(92,203,143,.13); --ci-bad:#F97066; --ci-bad-bg:rgba(249,112,102,.13);
  --ci-warn:#F2C46B; --ci-warn-bg:rgba(242,196,107,.10); --ci-warn-border:rgba(242,196,107,.30);
  --ci-shadow:0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.25);
"""

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { %(vars)s }

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}
.stApp { background: var(--ci-bg); }

/* hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { color: var(--ci-ink); letter-spacing: -0.015em; }
h1 { font-weight: 700 !important; }
p, li { line-height: 1.6; }

/* native bordered containers become soft cards */
[class*="st-key-card"] {
  background: var(--ci-surface); border: 1px solid var(--ci-border) !important;
  border-radius: 14px !important; box-shadow: var(--ci-shadow); padding: 1.1rem 1.25rem 1.2rem !important;
}

/* custom HTML cards */
.ci-card { background: var(--ci-surface); border: 1px solid var(--ci-border); border-radius: 14px;
  box-shadow: var(--ci-shadow); padding: 20px 22px; height: 100%%; color: var(--ci-ink); }
.ci-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; font-weight: 650; }
.ci-card p { margin: 0; color: var(--ci-muted); font-size: .93rem; }
.ci-card ul { margin: .3rem 0 0 1.1rem; padding: 0; color: var(--ci-muted); font-size: .93rem; }

.ci-kpi { background: var(--ci-surface); border: 1px solid var(--ci-border); border-radius: 14px;
  box-shadow: var(--ci-shadow); padding: 18px 20px; height: 100%%; }
.ci-kpi .label { font-size: .78rem; font-weight: 600; color: var(--ci-muted); text-transform: uppercase; letter-spacing: .06em; }
.ci-kpi .value { font-size: 1.85rem; font-weight: 700; color: var(--ci-ink); margin-top: 6px; line-height: 1.15; }
.ci-kpi .sub { font-size: .82rem; color: var(--ci-faint); margin-top: 4px; }
.ci-kpi .bar { width: 28px; height: 3px; border-radius: 3px; background: var(--ci-accent); margin-bottom: 12px; }

.ci-eyebrow { color: var(--ci-accent-ink); font-weight: 600; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .25rem; }
.ci-title { font-size: 2.05rem; font-weight: 700; color: var(--ci-ink); letter-spacing: -0.02em; margin: 0; line-height: 1.2; }
.ci-subtitle { color: var(--ci-muted); font-size: 1.02rem; margin: .45rem 0 1.4rem 0; max-width: 760px; }
.ci-section { margin: 1.9rem 0 .7rem 0; }
.ci-section h3 { font-size: 1.15rem; font-weight: 650; margin: 0; padding: 0; }
.ci-section p { color: var(--ci-muted); margin: .15rem 0 0 0; font-size: .92rem; }

.ci-hero { background: linear-gradient(135deg, var(--ci-accent-soft), transparent 70%%), var(--ci-surface);
  border: 1px solid var(--ci-border); border-radius: 16px; padding: 34px 36px; box-shadow: var(--ci-shadow); margin-bottom: 1.4rem; }
.ci-hero .ci-title { font-size: 2.5rem; }

.ci-card.step { min-height: 172px; }
.ci-step { display: flex; gap: 14px; align-items: flex-start; }
.ci-step .num { flex: 0 0 34px; height: 34px; border-radius: 50%%; background: var(--ci-accent-soft);
  color: var(--ci-accent-ink); font-weight: 700; display: flex; align-items: center; justify-content: center; }

.ci-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
  font-size: .82rem; font-weight: 600; }
.ci-badge.good { color: var(--ci-good); background: var(--ci-good-bg); }
.ci-badge.bad { color: var(--ci-bad); background: var(--ci-bad-bg); }
.ci-badge.neutral { color: var(--ci-accent-ink); background: var(--ci-accent-soft); }

.ci-test-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 22px; margin: 12px 0 14px 0; }
.ci-test-grid .k { font-size: .75rem; color: var(--ci-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.ci-test-grid .v { font-size: .95rem; color: var(--ci-ink); }
.ci-stat { font-size: 1.5rem; font-weight: 700; color: var(--ci-ink); }
.ci-conclusion { border-left: 3px solid var(--ci-accent); padding: 4px 0 4px 12px; color: var(--ci-ink); font-size: .95rem; }

.ci-disclaimer { background: var(--ci-warn-bg); border: 1px solid var(--ci-warn-border); border-radius: 12px;
  padding: 12px 16px; color: var(--ci-warn); font-size: .9rem; }
.ci-disclaimer b { color: var(--ci-warn); }

.ci-footer { margin-top: 2.6rem; padding-top: 1.1rem; border-top: 1px solid var(--ci-border);
  color: var(--ci-faint); font-size: .84rem; display: flex; flex-wrap: wrap; gap: 6px 22px; }
.ci-footer b { color: var(--ci-muted); font-weight: 600; }

/* widgets & tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--ci-border); }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; font-weight: 500; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stButton > button, .stFormSubmitButton > button { font-weight: 600; padding: .55rem 1.2rem; }
[data-testid="stSidebarNav"] a span { font-weight: 500; }

@media (max-width: 640px) {
  .ci-test-grid { grid-template-columns: 1fr; }
  .ci-hero { padding: 24px 20px; } .ci-hero .ci-title { font-size: 1.9rem; }
}
</style>
"""


def inject_css():
    st.markdown(_CSS % {"vars": _DARK if is_dark() else _LIGHT}, unsafe_allow_html=True)


# ---------------------------------------------------------------- components
def _key(*parts):
    """Build a widget key that is also a valid CSS class name."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", "-".join(map(str, parts)))[:100]


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow=None):
    eb = f'<div class="ci-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    html(f'{eb}<h1 class="ci-title">{escape(title)}</h1><p class="ci-subtitle">{subtitle}</p>')


def section(title, caption=None):
    cap = f"<p>{caption}</p>" if caption else ""
    html(f'<div class="ci-section"><h3>{escape(title)}</h3>{cap}</div>')


def kpi(label, value, sub=""):
    return (f'<div class="ci-kpi"><div class="bar"></div><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(str(value))}</div><div class="sub">{escape(sub)}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            html(kpi(label, value, sub))


def card(title, body):
    html(f'<div class="ci-card"><h4>{escape(title)}</h4>{body}</div>')


def badge(text, kind="neutral"):
    icon = {"good": "●", "bad": "●", "neutral": "●"}[kind]
    return f'<span class="ci-badge {kind}">{icon} {escape(text)}</span>'


def test_result(r, stat_label="Statistic"):
    """Render a hypothesis-test result dict from src.stats in a consistent card."""
    from src.stats import fmt_p
    sig = badge("Significant (reject H0)", "good") if r["significant"] else badge("Not significant (fail to reject H0)", "bad")
    with st.container(border=True, key=_key("card-test", r["test"], r["h0"])):
        html(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div style="font-weight:650;color:var(--ci-ink)">{escape(r['test'])}</div>{sig}
        </div>
        <div class="ci-test-grid">
          <div><div class="k">Null hypothesis H0</div><div class="v">{escape(r['h0'])}</div></div>
          <div><div class="k">Alternative H1</div><div class="v">{escape(r['h1'])}</div></div>
          <div><div class="k">{escape(stat_label)}</div><div class="ci-stat">{r['statistic']:.3f}</div></div>
          <div><div class="k">p-value <span style="text-transform:none">(α = {r['alpha']})</span></div><div class="ci-stat">{fmt_p(r['p_value'])}</div></div>
        </div>
        <div class="ci-conclusion">{escape(r['conclusion'])}</div>
        """)


def disclaimer():
    html('<div class="ci-disclaimer"><b>Medical disclaimer.</b> CardioInsight is a student project for learning data science. '
         'It is <b>not</b> a medical device and does not give medical advice or a diagnosis. Its predictions can be wrong. '
         'Always consult a qualified doctor about heart health.</div>')


def footer():
    a = AUTHOR
    html(f'<div class="ci-footer"><span><b>{escape(a["name"])}</b></span><span>Roll no. {escape(a["roll"])}</span>'
         f'<span>Class/Batch {escape(a["batch"])}</span><span>Guide: {escape(a["guide"])}</span>'
         f'<span>{escape(a["dept"])}</span><span>Data Science Mini Project (CA2)</span></div>')


def plotly(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})


def missing_file(what, notebook):
    with st.container(border=True, key=_key("card-missing", what)):
        st.markdown(f"#### {what} not found")
        st.markdown(f"This page needs a file created by **{notebook}**. Run the notebooks in order "
                    f"(see README → *How to run*), then reload this page.")
        st.code(f"jupyter nbconvert --to notebook --execute --inplace notebooks/{notebook}", language="bash")


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def load_raw():
    return data.load_raw()


@st.cache_data(show_spinner=False)
def load_clean():
    return data.load_clean()


@st.cache_resource(show_spinner=False)
def load_model():
    if not C.MODEL_FILE.exists():
        return None
    return joblib.load(C.MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not C.METRICS_FILE.exists():
        return None
    return load_json(C.METRICS_FILE)


def require_data():
    if not C.RAW_DATA.exists():
        with st.container(border=True, key="card-missing-data"):
            st.markdown("#### Dataset not found")
            st.markdown("Expected `data/heart.csv`. Please put the Kaggle *Heart Failure Prediction* CSV there.")
        st.stop()
