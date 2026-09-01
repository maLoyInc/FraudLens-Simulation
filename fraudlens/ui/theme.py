"""Design system loading.

Injects the single dark stylesheet once per session and sets the page shell.
There is no light theme and no theme toggle: FraudLens is dark only.
"""

from __future__ import annotations

import streamlit as st

from ..core import config as cfg

CSS_PATH = cfg.UI_STATIC_DIR / "fraudlens.css"


def configure_page() -> None:
    st.set_page_config(
        page_title=cfg.APP_TITLE,
        page_icon=str(cfg.UI_STATIC_DIR / "favicon.png"),
        layout="centered",
        initial_sidebar_state="collapsed",
    )


@st.cache_data(show_spinner=False)
def _read_css() -> str:
    if not CSS_PATH.exists():
        raise FileNotFoundError(f"stylesheet missing: {CSS_PATH}")
    return CSS_PATH.read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)
