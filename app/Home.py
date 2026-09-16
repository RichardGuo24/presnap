"""PreSnap frontend — landing page.

    streamlit run app/Home.py

Needs the Postgres DB running (docker compose up -d db) and a trained model
(artifacts/gbm.joblib). Every number is served through presnap.inference, which
reuses the training feature code — so the app can't drift from the model.
"""

import streamlit as st

st.set_page_config(page_title="PreSnap", page_icon="🏈", layout="wide")

st.title("🏈 PreSnap")
st.subheader("NFL win probability from pre-snap game state")

st.markdown(
    """
Every number here is computed from what's known **before the snap** — score,
time, down, field position, the betting line — and never from what the play did.
A gradient-boosted model, trained on 2010–2022 and graded on the untouched
2023–24 seasons.

**Use the sidebar →**
- **Game** — pick any game and watch its win-probability curve swing.
- **Rankings** — a season's wildest games, ranked by total probability swing.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Seasons", "2010–2024")
c2.metric("Plays modeled", "605,941")
c3.metric("Test Brier", "0.151", help="Coin-flip baseline is 0.25. Lower is better.")
