"""Plain-English tour of the terms and the math — an embedded interactive."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import presnap

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from presnap import webui  # noqa: E402

st.set_page_config(page_title="PreSnap · Explained", page_icon="🏈", layout="wide")
webui.inject()

html = (Path(__file__).resolve().parents[1] / "how_it_works.html").read_text()
components.html(html, height=2000, scrolling=True)
