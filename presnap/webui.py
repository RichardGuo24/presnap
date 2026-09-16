"""Shared visual identity for the Streamlit app.

Kept here (not in app/) because pages already import `presnap` reliably. Pages
call inject() once near the top. Pure strings + one thin st.markdown call.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --brand: #1A7A4C; --brand-ink: #0F5537; --amber: #E0902A;
  --ink: #171A1F; --muted: #5B6169; --paper: #F7F7F4; --line: #E6E6E0;
}

/* typography — replace Streamlit's default face entirely */
.stApp, .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"], .stSelectbox, .stDataFrame {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', system-ui, sans-serif;
  letter-spacing: -.015em; color: var(--ink); font-weight: 700; }
h1 { letter-spacing: -.03em; }
a { color: var(--brand); }

/* editorial column + spacing */
.block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 4rem; }

/* brand rule */
.presnap-brand { display: flex; align-items: baseline; gap: 10px;
  border-bottom: 2px solid var(--brand); padding-bottom: 8px; margin-bottom: 22px; }
.presnap-brand .name { font-family: 'Space Grotesk'; font-weight: 700;
  font-size: 1.05rem; letter-spacing: .04em; color: var(--ink); }
.presnap-brand .tag { font-size: .82rem; color: var(--muted); }
.presnap-brand .dot { width: 9px; height: 9px; border-radius: 50%;
  background: var(--brand); align-self: center; }

/* dark branded sidebar */
section[data-testid="stSidebar"] { background: #12211A; border-right: 1px solid #22382E; }
section[data-testid="stSidebar"] * { color: #DCE6E0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 8px; }
section[data-testid="stSidebar"] [aria-current="page"] { background: rgba(26,122,76,.30) !important; }

/* metric tiles */
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
  border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk'; font-weight: 700; }

/* misc polish */
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stExpander"] summary { font-weight: 600; }
footer { visibility: hidden; height: 0; }
</style>
"""

BRAND = (
    '<div class="presnap-brand"><span class="dot"></span>'
    '<span class="name">PRESNAP</span>'
    '<span class="tag">NFL win probability</span></div>'
)


def inject() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(BRAND, unsafe_allow_html=True)
