from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bank_marketing.ui.pages import render_app

st.set_page_config(
    page_title="DepoRank | Campaign Analytics",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_app(PROJECT_ROOT)
