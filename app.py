# app.py

import streamlit as st
from stock_pages import swiss_market_assets_page, variance_check_page

# Page configuration
st.set_page_config(page_title="Stock App", layout="wide")

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page", 
    ["Swiss Market Assets Overview", "Variance Check"]
)

if page == "Swiss Market Assets Overview":
    swiss_market_assets_page()
elif page == "Variance Check":
    variance_check_page()

