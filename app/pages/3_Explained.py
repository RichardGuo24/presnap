"""Plain-English tour of the terms and the math — an embedded interactive."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PreSnap · Explained", page_icon="🏈", layout="wide")

html = (Path(__file__).resolve().parents[1] / "how_it_works.html").read_text()
components.html(html, height=2000, scrolling=True)
