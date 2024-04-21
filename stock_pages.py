# stock_pages.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import plotly.express as px
import base64
from openai import OpenAI
from datetime import datetime
from functools import lru_cache
from matplotlib.dates import DateFormatter
from constants import (BASE_URL, HISTORICAL_PRICE_URL, SWISS_STOCKS, 
                       SWISS_MARKET_ASSETS, STOCK_THRESHOLD, ETF_THRESHOLD)
from keys import API_KEY, OPENAI_API_KEY
import streamlit as st

client = OpenAI(
    api_key= OPENAI_API_KEY,
    organization="org-tlUjsA0VUZEicWZ3HTdlVHJH",
    project="proj_TVJJRleVFZjzvzzMol4NWKPq"
)


@st.cache_resource(ttl=3600)
def fetch_stock_data(symbol):
    try:
        response = requests.get(f"{BASE_URL}quote/{symbol}?apikey={API_KEY}")
        response.raise_for_status()
        data = response.json()
        if not data:
            st.warning(f"No data available for {symbol}.")
            return None
        return {'price': data[0]['price'], 'previousClose': data[0]['previousClose'], 'fullName': data[0]['name']}  # Assuming 'name' is the key for full name
    except requests.RequestException as e:
        st.error(f"Failed to retrieve data: {e}")
        return None


@st.cache_resource(ttl=3600)
def fetch_historical_data(symbol):
    url = f"{BASE_URL}historical-price-full/{symbol}?timeseries=180&apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['historical']
        prices = [day['close'] for day in data if 'close' in day]
        dates = [day['date'] for day in data if 'date' in day]
        return prices, dates
    except requests.RequestException as e:
        st.error(f"Failed to retrieve historical data for {symbol}: {e}")
        return [], []
    
def get_chatgpt_summary(query):
    """
    Function to get summaries from the ChatGPT API using GPT-4.
    Queries recent news based on the percentage change of a financial product.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=150
        )
        
        # Accessing the first message content if available
        # Using proper attribute access for Pydantic models
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()  # Corrected access here
        else:
            return "No summary available."
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return "Failed to retrieve summary due to an API error."
 
    

def calculate_std_dev(prices):
    """Calculate the standard deviation of the provided price list."""
    if not prices:
        return 0, 0  # Return mean as 0 for consistency
    return np.mean(prices), np.std(prices)

def calculate_modified_z_scores(prices):
    median = np.median(prices)
    deviations = [abs(x - median) for x in prices]
    mad = np.median(deviations)

    if mad == 0:
        # If MAD is zero, all prices are identical, return zero Z-scores indicating no outliers
        return [0] * len(prices)

    return [0.6745 * (x - median) / mad for x in prices]

def outlier_detection(today_price, historical_prices):
    all_prices = historical_prices + [today_price['price']]
    historical_z_scores = calculate_modified_z_scores(historical_prices)
    all_z_scores = calculate_modified_z_scores(all_prices)

    # Last Z-score corresponds to today's price
    today_z_score = all_z_scores[-1]
    threshold_z = 3.5

    # Check if today's price is an outlier
    is_outlier = abs(today_z_score) > threshold_z
    return is_outlier, all_z_scores[-2:], historical_z_scores

def initialize_app():
    st.title("Variance Check for Assets")
    st.sidebar.write("This page checks if the asset's current price change is above a threshold and compares today's variance against historical variance.")
    asset_symbol = st.sidebar.text_input("Enter a ticker symbol (e.g., 'AAPL'):", '')

    if asset_symbol:
        selected_asset = {'symbol': asset_symbol.upper(), 'type': 'Stock'}  # Default to Stock for simplicity
    else:
        selected_asset = st.sidebar.selectbox('Or select an Asset', SWISS_MARKET_ASSETS, format_func=lambda x: x['symbol'])

    return selected_asset

def process_asset_data(selected_asset):
    threshold = ETF_THRESHOLD if selected_asset['type'] == 'ETF' else STOCK_THRESHOLD
    historical_prices, historical_dates = fetch_historical_data(selected_asset['symbol'])
    today_price_data = fetch_stock_data(selected_asset['symbol'])
    
    if not today_price_data or not historical_prices:
        st.error("Failed to fetch today's price or historical data for the selected asset.")
        return None  # Early exit if data fetch fails
    
    today_price = today_price_data['price']
    previous_close = today_price_data['previousClose']
    today_change = ((today_price - previous_close) / previous_close) * 100  
    
    historical_mean, historical_std_dev = calculate_std_dev(historical_prices)
    
    return (today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold)

def plot_historical_prices(historical_dates, historical_prices):
    df = pd.DataFrame({'Date': pd.to_datetime(historical_dates), 'Price': historical_prices})
    fig = px.line(df, x='Date', y='Price', title='Historical Price Development (Last 90 Days)',
                  labels={'Price': 'Closing Price ($)'}).update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_variance_comparison(historical_mean, historical_std_dev, today_mean, today_std_dev):
    fig, ax = plt.subplots()
    x_range = np.linspace(min(historical_mean - 3*historical_std_dev, today_mean - 3*today_std_dev),
                          max(historical_mean + 3*historical_std_dev, today_mean + 3*today_std_dev), 100)
    ax.plot(x_range, norm.pdf(x_range, historical_mean, historical_std_dev), label='Historical', color='blue')
    ax.plot(x_range, norm.pdf(x_range, today_mean, today_std_dev), label='Today', color='red', linestyle='dashed')
    ax.fill_between(x_range, norm.pdf(x_range, historical_mean, historical_std_dev), norm.pdf(x_range, today_mean, today_std_dev),
                    where=(norm.pdf(x_range, today_mean, today_std_dev) > norm.pdf(x_range, historical_mean, historical_std_dev)),
                    facecolor='green', alpha=0.3, label='Increased Volatility (Today > Historical)')
    ax.fill_between(x_range, norm.pdf(x_range, historical_mean, historical_std_dev), norm.pdf(x_range, today_mean, today_std_dev),
                    where=(norm.pdf(x_range, today_mean, today_std_dev) <= norm.pdf(x_range, historical_mean, historical_std_dev)),
                    facecolor='orange', alpha=0.3, label='Decreased Volatility (Today <= Historical)')
    ax.set_title('Price Distribution Comparison')
    ax.set_ylabel('Probability Density')
    ax.legend()
    st.pyplot(fig)

def plot_price_distribution(historical_mean, historical_std_dev, today_price):
    # Generate data for a standard normal curve
    x_range = np.linspace(historical_mean - 3*historical_std_dev, historical_mean + 3*historical_std_dev, 400)
    y_range = stats.norm.pdf(x_range, historical_mean, historical_std_dev)

    fig, ax = plt.subplots()
    ax.plot(x_range, y_range, label='Historical Price Distribution', color='blue')

    # Highlight the area under the curve where today's price falls
    ax.fill_between(x_range, 0, y_range, where=(x_range <= today_price), color='green', alpha=0.5)

    # Add vertical line for today's price that stops at the curve
    today_price_point = stats.norm.pdf(today_price, historical_mean, historical_std_dev)
    ax.plot([today_price, today_price], [0, today_price_point], color='red', linestyle='dashed', label=f"Today's Price = {today_price}")

    # Add vertical line for average price that stops at the curve
    average_price_point = stats.norm.pdf(historical_mean, historical_mean, historical_std_dev)
    ax.plot([historical_mean, historical_mean], [0, average_price_point], color='yellow', linestyle='dashed', label=f"90-Day Average Price = {historical_mean}")

    # Adjust legend location to 'upper right'
    ax.legend(loc='upper right')

    ax.set_title('Normal Distribution of Historical Prices')
    ax.set_xlabel('Price')
    ax.set_ylabel('Probability Density')

    st.pyplot(fig)

def display_results(today_price, today_change, historical_mean, historical_std_dev,
                    historical_dates, historical_prices, threshold, today_price_data, selected_asset):
    # Calculate Z-scores for all involved prices
    is_outlier, today_z_scores, historical_z_scores = outlier_detection(today_price_data, historical_prices)

    with st.expander("Historical Price Development"):
        plot_historical_prices(historical_dates, historical_prices)
    
    with st.expander("Price Distribution Comparison"):
        plot_price_distribution(historical_mean, historical_std_dev, today_price)

    # Check if today's price change is an outlier
    today_is_outlier = abs(today_z_scores[-1]) > 3.5

    # Create DataFrame for results 
    results_df = pd.DataFrame({
        "Symbol": [selected_asset['symbol']],
        "Full Name": [today_price_data['fullName']],
        "Today's Price": [today_price],
        "Yesterday's Price": [today_price_data['previousClose']],
        "Percentage Change": [f"{today_change:.2f}%"],
        "Change Above Threshold": ["Yes" if abs(today_change) > threshold else "No"],
        "Today's Z-Score": [f"{today_z_scores[-1]:.2f}"],
        "Outlier Status": ["Yes" if today_is_outlier else "No"]
    })
    
    # Convert DataFrame to CSV for download
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{selected_asset["symbol"]}_price_results.csv">Download Results as CSV</a>'
    
    # Display DataFrame and download link
    st.write(results_df)
    st.markdown(href, unsafe_allow_html=True)
    
    # Display messages based on the threshold check and z-score analysis
    if today_is_outlier:
        st.error(f"Today's price change is an outlier with a Z-score of {today_z_scores[-1]:.2f}.")
    else:
        st.success("Today's price change is not considered an outlier based on Z-scores.")

    if abs(today_change) > threshold:
        st.success(f"The price change today is {today_change:.2f}%, which is above the {threshold}% threshold.")
    else:
        st.error(f"The price change today is {today_change:.2f}%, which is below the {threshold}% threshold. This might indicate less market activity or stability depending on the context.")

def display_stock_data(stock_list):
    results = []
    for asset in stock_list:
        symbol = asset['symbol']
        type_asset = asset['type']
        
        data = fetch_stock_data(symbol)
        if data:
            previous_close = data['previousClose']
            price = data['price']
            change_percent = ((price - previous_close) / previous_close) * 100
            threshold = ETF_THRESHOLD if type_asset == 'ETF' else STOCK_THRESHOLD
            above_threshold = "Above" if abs(change_percent) > threshold else "Below"
            results.append([symbol, type_asset, f"${price:.2f}", f"${previous_close:.2f}", f"{change_percent:.2f}%", above_threshold])
        else:
            results.append([symbol, type_asset, "N/A", "N/A", "N/A", "N/A"])

    df = pd.DataFrame(results, columns=['Symbol', 'Type', 'Price', 'Previous Close', 'Change %', 'Threshold Status'])
    st.table(df)

    # Convert DataFrame to CSV for download
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="Swiss_Market_Assets.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def variance_check_page():
    selected_asset = initialize_app()
    result = process_asset_data(selected_asset)
    
    if result:
        today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold = result
        today_price_data = fetch_stock_data(selected_asset['symbol'])
        
        if today_price_data:
            display_results(today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold, today_price_data, selected_asset)
            
            if st.checkbox("Generate Summary of Relevant News"):
                st.write("Checkbox is checked, generating summary...")
                news_query = f"Provide a summary of recent news related to the {today_price_data['fullName']} ({selected_asset['symbol']}) which experienced a {today_change:.2f}% change yesterday."
                
                summary = get_chatgpt_summary(news_query)
                st.write("### Summary of Relevant News")
                st.write(summary)
                
                print("Sending query to API:", news_query)
        else:
            st.error("Failed to retrieve today's price data.")
    else:
        st.error("Failed to process asset data.")

def swiss_market_assets_page():
    st.title("Overview of Swiss Market Assets")
    display_stock_data(SWISS_MARKET_ASSETS)


