# stock_pages.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import base64
from openai import OpenAI
from datetime import datetime
from functools import lru_cache
from matplotlib.dates import DateFormatter
from constants import (BASE_URL, HISTORICAL_PRICE_URL, SWISS_STOCKS, 
                       SWISS_MARKET_ASSETS, STOCK_THRESHOLD, ETF_THRESHOLD)
#from keys import API_KEY, OPENAI_API_KEY
import streamlit as st
import random

client = OpenAI(
    api_key=st.secrets["api_keys"]["OPENAI_API_KEY"],
    organization="org-tlUjsA0VUZEicWZ3HTdlVHJH",
    project="proj_TVJJRleVFZjzvzzMol4NWKPq"
)


@st.cache_resource(ttl=3600)
def fetch_stock_data(symbol):
    try:
        # Access API key from secrets
        api_key = st.secrets["api_keys"]["API_KEY"]
        response = requests.get(f"{BASE_URL}quote/{symbol}?apikey={api_key}")
        response.raise_for_status()
        data = response.json()
        if not data:
            st.warning(f"No data available for {symbol}.")
            return None
        return {'price': data[0]['price'], 'previousClose': data[0]['previousClose'], 'fullName': data[0]['name']}
    except requests.RequestException as e:
        st.error(f"Failed to retrieve data: {e}")
        return None


@st.cache_resource(ttl=3600)
def fetch_historical_data(symbol):
    try:
        # Access API key from secrets
        api_key = st.secrets["api_keys"]["API_KEY"]
        url = f"{BASE_URL}historical-price-full/{symbol}?timeseries=180&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['historical']

        # Sort data by date in ascending order to ensure oldest data comes first
        sorted_data = sorted(data, key=lambda x: x['date'])

        prices = [day['close'] for day in sorted_data if 'close' in day]
        dates = [day['date'] for day in sorted_data if 'date' in day]
        return prices, dates
    except requests.RequestException as e:
        st.error(f"Failed to retrieve historical data for {symbol}: {e}")
        return [], []

    
def fetch_news(symbol):
    """Fetch the latest news for a given stock symbol using FMP API."""
    api_key = st.secrets["api_keys"]["API_KEY"]
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit=5&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a list of news articles
    else:
        st.error("Failed to fetch news")
        return []
    
def fetch_news_for_stocks(stock_list):
    """Fetch news for a list of stocks using the Financial Modeling Prep API."""
    api_key = st.secrets["api_keys"]["API_KEY"]
    aggregated_news = []
    for stock in stock_list:
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={stock['symbol']}&limit=2&apikey={api_key}"  # Limit to 2 articles per stock
        response = requests.get(url)
        if response.status_code == 200:
            aggregated_news.extend(response.json())
        else:
            st.error(f"Failed to fetch news for {stock['symbol']}: {response.status_code}")
    return aggregated_news
    
def get_chatgpt_summary(news_articles, today_change, full_name, symbol):
    """
    Function to get summaries from the ChatGPT API using GPT-4. Combines news articles into a summary.
    """
    news_text = ' '.join([article['text'] for article in news_articles])  # Combine texts of all articles
    query = f"Summarize the following news related to {full_name} ({symbol}) which experienced a {today_change:.2f}% change yesterday: {news_text}"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=250  # Adjusted to allow a more comprehensive summary
        )
        
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()
        else:
            return "No summary available."
    except Exception as e:
        st.error(f"Failed to retrieve summary due to an API error: {str(e)}")
        return None

def summarize_market_behavior(news_articles):
    """Generate a summary of the day's market behavior using the ChatGPT API."""
    if not news_articles:
        return "No news articles available for summarization."
    
    news_text = ' '.join([article['text'] for article in news_articles])
    query = f"Provide a summary of today's market behavior based on the following news: {news_text}"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=300  # Increased token count for a broader summary
        )
        
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()
        else:
            return "No summary available."
    except Exception as e:
        st.error(f"Failed to retrieve summary due to an API error: {str(e)}")
        return None

def display_market_summary():
    st.write("## Market Overview")
    if st.checkbox("Generate Summary of Today's Market Behavior"):
        st.write("Fetching market news...")
        market_news = fetch_news_for_stocks(SWISS_STOCKS)
        if market_news:
            st.write("Generating summary...")
            market_summary = summarize_market_behavior(market_news)
            st.write("### Summary of Today's Market Behavior")
            st.write(market_summary)
        else:
            st.write("No recent market news found.")

def analyze_and_display_data(selected_asset, prices, dates, z_score_threshold):
    # Analysis (Outlier Detection etc.)
    mean_price, std_dev_price = calculate_std_dev(prices)
    is_outlier, today_z_scores, historical_z_scores = outlier_detection({'price': prices[-1]}, prices[:-1], z_score_threshold)

    # Displaying results
    plot_historical_prices(dates, prices)
    plot_price_distribution(mean_price, std_dev_price, prices[-1])

    # Display statistics and results
    st.write(f"Mean price over the last 30 days: {mean_price:.2f}")
    st.write(f"Standard deviation of prices over the last 30 days: {std_dev_price:.2f}")
    st.write(f"Today's Z-Score: {today_z_scores[-1]:.2f}")
    st.write("Outlier Status: Yes" if is_outlier else "Outlier Status: No")

def perform_daily_checks(prices, dates, z_score_threshold, price_change_threshold):
    results = []
    price_changes = []
    z_scores = []
    count_above_threshold = 0
    count_outliers = 0

    for i in range(1, len(prices)):
        today_price = prices[i]
        historical_prices = prices[:i]
        if not historical_prices:
            continue  # Skip the first day since there's no historical data

        mean_price, std_dev_price = calculate_std_dev(historical_prices)
        is_outlier, today_z_scores, _ = outlier_detection({'price': today_price}, historical_prices, z_score_threshold)
        today_change = ((today_price - prices[i-1]) / prices[i-1]) * 100 if prices[i-1] != 0 else 0

        price_changes.append(today_change)
        z_scores.append(today_z_scores[-1])

        change_above_threshold = "Yes" if abs(today_change) > price_change_threshold else "No"
        if change_above_threshold == "Yes":
            count_above_threshold += 1
        if is_outlier:
            count_outliers += 1
        
        results.append({
            'date': dates[i],
            'today_price': today_price,
            'price_change': today_change,
            'change_above_threshold': change_above_threshold,
            'z_score': today_z_scores[-1],
            'is_outlier': 'Yes' if is_outlier else 'No'
        })

    return results, count_above_threshold, count_outliers, price_changes, z_scores

def plot_changes_and_scores(dates, price_changes, z_scores, prices, price_change_threshold, z_score_threshold, altered_dates):
    fig = go.Figure()

    # Convert price_changes and z_scores to absolute values
    abs_price_changes = [abs(x) for x in price_changes]
    abs_z_scores = [abs(x) for x in z_scores]

    # Adjust the dates array for price_changes and z_scores to exclude the first date
    adjusted_dates = dates[1:]  # Starting from the second element to align with price_changes and z_scores

    # Add Price Change trace and its threshold line
    price_change_trace = go.Scatter(x=adjusted_dates, y=abs_price_changes, mode='lines+markers',
                                    name='Absolute Price Change (%)', line=dict(color='blue'),
                                    legendgroup='price_change', showlegend=True)
    price_threshold_line = go.Scatter(x=[adjusted_dates[0], adjusted_dates[-1]], y=[price_change_threshold, price_change_threshold],
                                      mode='lines', line=dict(color='blue', dash='dash', width=2),
                                      legendgroup='price_change', showlegend=False)

    # Add Z-Score trace and its threshold line
    z_score_trace = go.Scatter(x=adjusted_dates, y=abs_z_scores, mode='lines+markers',
                               name='Absolute Z-Score', yaxis='y2', line=dict(color='red'),
                               legendgroup='z_score', showlegend=True)
    z_score_threshold_line = go.Scatter(x=[adjusted_dates[0], adjusted_dates[-1]], y=[z_score_threshold, z_score_threshold],
                                        mode='lines', line=dict(color='red', dash='dash', width=2),
                                        legendgroup='z_score', showlegend=False, yaxis='y2')

    fig.add_traces([price_change_trace, price_threshold_line, z_score_trace, z_score_threshold_line])

    # Add Price Development trace
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price Development',
                             line=dict(dash='dash', color='green', width=2), yaxis='y3', showlegend=True))

    # Optional: Altered Days markers
    altered_prices = [price for date, price in zip(dates, prices) if date in altered_dates]
    altered_dates_prices = [date for date in dates if date in altered_dates]
    fig.add_trace(go.Scatter(x=altered_dates_prices, y=altered_prices, mode='markers',
                             name='Altered Days', marker=dict(color='red', size=10, symbol='circle'),
                             showlegend=True))

    # Layout adjustments
    fig.update_layout(
        title="Absolute Price Change, Absolute Z-Score, and Price Development Over the Last 30 Days",
        xaxis_title="Date",
        yaxis=dict(title='Absolute Price Change (%)', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Absolute Z-Score', titlefont=dict(color='red'), tickfont=dict(color='red'), anchor='x', overlaying='y', side='right'),
        yaxis3=dict(title='Price ($)', titlefont=dict(color='green'), tickfont=dict(color='green'), anchor='free', overlaying='y', side='right', position=0.95),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    

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

def outlier_detection(today_price, historical_prices, z_score_threshold):
    all_prices = historical_prices + [today_price['price']]
    historical_z_scores = calculate_modified_z_scores(historical_prices)
    all_z_scores = calculate_modified_z_scores(all_prices)

    # Last Z-score corresponds to today's price
    today_z_score = all_z_scores[-1]

    # Check if today's price is an outlier based on user-defined Z-score threshold
    is_outlier = abs(today_z_score) > z_score_threshold
    return is_outlier, all_z_scores[-2:], historical_z_scores

def simulate_false_data_points(prices, dates):
    simulated_prices = []
    altered_dates = []
    for price, date in zip(prices, dates):
        if random.random() < 0.05:  # % chance to alter the data point
            factor = random.uniform(1.5, 3)
            if random.random() < 0.5:
                simulated_prices.append(price * factor)  # Multiply by the factor
            else:
                simulated_prices.append(price / factor)  # Divide by the factor
            altered_dates.append(date)  # Track the date of alteration
        else:
            simulated_prices.append(price)
    return simulated_prices, altered_dates

def analyze_and_count_alerts(results, altered_dates):
    correct_price_threshold_alerts = 0
    correct_z_score_outliers = 0

    # Convert altered_dates to a set for faster lookup
    altered_dates_set = set(altered_dates)

    # Iterate through results and check for correct detections
    for result in results:
        if result['date'] in altered_dates_set:
            if result['change_above_threshold'] == 'Yes':
                correct_price_threshold_alerts += 1
            if result['is_outlier'] == 'Yes':
                correct_z_score_outliers += 1

    return correct_price_threshold_alerts, correct_z_score_outliers


def initialize_app():
    st.sidebar.write("This page checks if the asset's current price change is above a threshold and compares today's variance against historical variance.")

    asset_symbol = st.sidebar.text_input("Enter a ticker symbol (e.g., 'AAPL'):", '')
    if asset_symbol:
        selected_asset = {'symbol': asset_symbol.upper(), 'type': 'Stock'}  # Default type as 'Stock'
    else:
        selected_asset = st.sidebar.selectbox('Or select an Asset', SWISS_MARKET_ASSETS, format_func=lambda x: x['symbol'])

    # Dynamic threshold sliders based on asset type
    if selected_asset.get('type', 'Stock') == 'ETF':
        price_change_threshold = st.sidebar.slider("Define ETF price change threshold (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    else:
        price_change_threshold = st.sidebar.slider("Define Stock price change threshold (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    z_score_threshold = st.sidebar.slider("Set Z-score threshold for outlier detection", min_value=0.0, max_value=5.0, value=2.5, step=0.1)

    return selected_asset, price_change_threshold, z_score_threshold

def process_asset_data(selected_asset, threshold):
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
                    historical_dates, historical_prices, price_change_threshold, z_score_threshold, 
                    today_price_data, selected_asset):
    # Calculate Z-scores for all involved prices
    is_outlier, today_z_scores, historical_z_scores = outlier_detection(today_price_data, historical_prices, z_score_threshold)

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Historical Price Development"):
            plot_historical_prices(historical_dates, historical_prices)

    with col2:
        with st.expander("Price Distribution"):
            plot_price_distribution(historical_mean, historical_std_dev, today_price)

    results_df = pd.DataFrame({
        "Symbol": [selected_asset['symbol']],
        "Full Name": [today_price_data['fullName']],
        "Today's Price": [today_price],
        "Yesterday's Price": [today_price_data['previousClose']],
        "Percentage Change": [f"{today_change:.2f}%"],
        "Change Above Threshold": ["Yes" if abs(today_change) > price_change_threshold else "No"],
        "Today's Z-Score": [f"{today_z_scores[-1]:.2f}"],
        "Outlier Status": ["Yes" if is_outlier else "No"]
    })

    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{selected_asset["symbol"]}_price_results.csv">Download Results as CSV</a>'
    st.write(results_df)
    st.markdown(href, unsafe_allow_html=True)

    if is_outlier:
        st.error(f"Today's price change is an outlier with a Z-score of {today_z_scores[-1]:.2f}.")
    else:
        st.success("Today's price change is not considered an outlier based on Z-scores.")

    if abs(today_change) > price_change_threshold:
        st.success(f"The price change today is {today_change:.2f}%, which is above the {price_change_threshold}% threshold.")
    else:
        st.error(f"The price change today is {today_change:.2f}%, which is below the {price_change_threshold}% threshold. This might indicate less market activity or stability depending on the context.")

    

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

def display_analysis_results(results, count_above_threshold, count_outliers, altered_dates, price_change_threshold, z_score_threshold):
    # Initialize counts for altered days identified above respective thresholds
    count_altered_price_change_above_threshold = 0
    count_altered_z_score_outliers = 0

    # Count altered days that are correctly identified above the thresholds
    for result in results:
        if result['date'] in altered_dates:
            if abs(result['price_change']) > price_change_threshold:
                count_altered_price_change_above_threshold += 1
            if abs(result['z_score']) > z_score_threshold:
                count_altered_z_score_outliers += 1

    # Prepare data for display
    metrics = ['Price Change Above Threshold', 'Z-Score Outliers']
    counts = [count_above_threshold, count_outliers]
    descriptions = ['Total days with price change above threshold', 'Total days with Z-Score outliers']

    if altered_dates:
        metrics.extend(['Altered Days Price Change Detected', 'Altered Days Z-Score Detected'])
        counts.extend([count_altered_price_change_above_threshold, count_altered_z_score_outliers])
        descriptions.extend(['Altered days correctly identified above price change threshold',
                             'Altered days correctly identified above Z-Score threshold'])

    results_data = {
        'Metric': metrics,
        'Count': counts,
        'Description': descriptions
    }

    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame(data=results_data)
    
    # Display the DataFrame as a table in Streamlit
    st.table(results_df)

def variance_check_page():
    st.title("Detailed Analysis")
    selected_asset, price_change_threshold, z_score_threshold = initialize_app()
    result = process_asset_data(selected_asset, price_change_threshold)
    
    if result:
        today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, threshold = result
        today_price_data = fetch_stock_data(selected_asset['symbol'])
        
        if today_price_data:
            display_results(today_price, today_change, historical_mean, historical_std_dev, historical_dates, historical_prices, price_change_threshold, z_score_threshold, today_price_data, selected_asset)
            
            with st.expander("Generate News Summary"):
                # Checkbox for market behavior summary
                """
                if st.checkbox("Generate Summary of Today's Market Behavior"):
                    st.write("Fetching market news...")
                    market_news = fetch_news_for_stocks(SWISS_STOCKS)  # Assuming this fetches aggregated news for the stocks
                    if market_news:
                        st.write("Generating summary...")
                        market_summary = summarize_market_behavior(market_news)
                        st.write("### Summary of Today's Market Behavior")
                        st.write(market_summary)
                    else:
                        st.write("No recent market news found.")
                        """
                
                # Checkbox for specific stock news summary (only works for american Stocks/ETFs)
                if st.checkbox("(Only for American Stocks/ETFs) Generate Summary of Relevant News for " + selected_asset['symbol']):
                    news_articles = fetch_news(selected_asset['symbol'])
                    if news_articles:
                        st.write("Generating summary...")
                        summary = get_chatgpt_summary(news_articles, today_change, today_price_data['fullName'], selected_asset['symbol'])
                        st.write("### Summary of Relevant News")
                        st.write(summary)
                    else:
                        st.write("No recent news articles found.")
        else:
            st.error("Failed to retrieve today's price data.")
    else:
        st.error("Failed to process asset data.")

def last_30_days_analysis_page():
    st.title("30-Day Analysis")

    selected_asset, price_change_threshold, z_score_threshold = initialize_app()
    historical_prices, historical_dates = fetch_historical_data(selected_asset['symbol'])
    
    if not historical_prices or len(historical_prices) < 30:
        st.error("Insufficient data for a full 30-day analysis.")
        return

    simulate_data = st.checkbox("Simulate false data points")
    recent_30_days_prices = historical_prices[-30:]
    recent_30_days_dates = historical_dates[-30:]

    altered_dates = []
    if simulate_data:
        recent_30_days_prices, altered_dates = simulate_false_data_points(recent_30_days_prices, recent_30_days_dates)

    results, count_above_threshold, count_outliers, price_changes, z_scores = perform_daily_checks(recent_30_days_prices, recent_30_days_dates, z_score_threshold, price_change_threshold)

    plot_changes_and_scores(recent_30_days_dates, price_changes, z_scores, recent_30_days_prices, price_change_threshold, z_score_threshold, altered_dates)

    display_analysis_results(results, count_above_threshold, count_outliers, altered_dates, price_change_threshold, z_score_threshold)

    if simulate_data:
        # Additional stats for simulated data points
        correct_price_alerts, correct_z_score_outliers = analyze_and_count_alerts(results, altered_dates)
        st.write(f"Out of the {len(altered_dates)} altered data points, {correct_price_alerts} were correctly identified as exceeding the price change threshold.")
        st.write(f"Out of the {len(altered_dates)} altered data points, {correct_z_score_outliers} were correctly identified as Z-score outliers.")

    with st.expander("Show Results"):
        results_df = pd.DataFrame(results)
        results_df = results_df[['date', 'today_price', 'price_change', 'change_above_threshold', 'z_score', 'is_outlier']]
        st.table(results_df)

        # Optionally, allow downloading the results
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{selected_asset["symbol"]}_30_day_analysis.csv">Download 30-Day Analysis Results</a>'
        st.markdown(href, unsafe_allow_html=True)





def swiss_market_assets_page():
    st.title("Overview of Swiss Market Assets")
    display_stock_data(SWISS_MARKET_ASSETS)


