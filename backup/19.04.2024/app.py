# app.py

import streamlit as st
from stock_pages import stock_search_page, swiss_stocks_overview_page, swiss_market_assets_page, variance_check_page

# Page configuration
st.set_page_config(page_title="Stock App", layout="wide")

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page", 
    ["Stock Search", "Swiss Stocks Overview", "Swiss Market Assets Overview", "Variance Check"]
)

if page == "Stock Search":
    stock_search_page()
elif page == "Swiss Stocks Overview":
    swiss_stocks_overview_page()
elif page == "Swiss Market Assets Overview":
    swiss_market_assets_page()
elif page == "Variance Check":
    variance_check_page()

