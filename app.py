import streamlit as st

# Page configuration  
st.set_page_config(
    page_title="Ogelo SIMS - Smart Infrastructure Monitoring",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirect to Home page
st.switch_page("pages/0_🏠_Home.py")