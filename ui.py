"""Streamlit UI for the research agent:  streamlit run ui.py

Two pages in the sidebar: Researcher (the chat, app_pages/researcher.py) and Dashboard (the local log
dashboard, app_pages/dashboard.py).
"""
import streamlit as st

st.set_page_config(page_title="ML Researcher", page_icon="🔬", layout="wide")

# The sidebar's own collapse button («) normally shows only while the mouse is over the sidebar header;
# keep it visible. Once collapsed, Streamlit's expand button (») at the top left opens it again.
st.html("""<style>
[data-testid="stSidebarCollapseButton"] { display: inline-flex !important; visibility: visible !important;
                                          opacity: 1 !important; }
</style>""")

st.navigation([
    st.Page("app_pages/researcher.py", title="Researcher", icon="🔬", default=True),
    st.Page("app_pages/dashboard.py", title="Dashboard", icon="📈"),
]).run()
