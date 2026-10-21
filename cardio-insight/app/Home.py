"""CardioInsight - entry point.

Run from the project root:
    streamlit run app/Home.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CardioInsight", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="expanded")

import ui  # noqa: E402  (must come after set_page_config)

ui.inject_css()

pages = {
    "Overview": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/data_explorer.py", title="Data Explorer", icon=":material/table_view:"),
    ],
    "Statistics": [
        st.Page("views/statistical_tests.py", title="Statistical Tests", icon=":material/functions:"),
        st.Page("views/correlations.py", title="Correlations", icon=":material/grid_on:"),
        st.Page("views/outliers.py", title="Outliers", icon=":material/scatter_plot:"),
    ],
    "Model": [
        st.Page("views/model_performance.py", title="Model Performance", icon=":material/insights:"),
        st.Page("views/risk_predictor.py", title="Risk Predictor", icon=":material/favorite:"),
    ],
    "About": [
        st.Page("views/ethics_about.py", title="Ethics & About", icon=":material/balance:"),
    ],
}

ASSETS = Path(__file__).parent / "assets"
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

st.navigation(pages).run()
