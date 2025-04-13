import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import time

# Load the model
model = load_model('Stock_Predictions_Model.h5')

# Custom Styles for Modern Background
def apply_styles():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to bottom, #ffffff, #f0f4f8);
            font-family: 'Arial', sans-serif;
            color: #333333;
        }
        .main-title {
            font-size: 36px;
            color: #1d3557;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .section-title {
            font-size: 26px;
            color: #457b9d;
            margin-top: 20px;
            font-weight: bold;
        }
        .success-box {
            background-color: #28a745;
            color: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #155724;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #ffc107;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_styles()

# Function to safely fetch data with retry mechanism
def fetch_data(ticker, start, end, retries=5, delay=10):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                return data
        except Exception:
            print(f"Rate limit exceeded for {ticker}. Retrying in {delay} seconds... ({attempt + 1}/{retries})")
            time.sleep(delay)
    st.markdown(f'<div class="warning-box">Failed to fetch data for <b>{ticker}</b> after multiple attempts.</div>', unsafe_allow_html=True)
    return pd.DataFrame()

# Streamlit app header
st.markdown('<div class="main-title">Enhanced Stock Market Predictor</div>', unsafe_allow_html=True)

# Section: Predict Tomorrow's Stock Price
st.markdown('<div class="section-title">Predict Tomorrow\'s Stock Price</div>', unsafe_allow_html=True)
stock = st.text_input('Enter Stock Symbol for Prediction', '^NSEI')

start = '2012-01-01'
end = date.today().strftime('%Y-%m-%d')
data = fetch_data(stock, start=start, end=end)

if not data.empty and len(data) > 1:
    st.subheader('Stock Data (Until Today)')
    st.write(data)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare input for tomorrow's prediction
    last_100_days = data_scaled[-100:]
    X_test = np.array([last_100_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict tomorrow's price
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    st.markdown(
        f'<div class="success-box">Tomorrow\'s Predicted Price for <b>{stock}</b>: {predicted_price[0][0]:.2f}</div>',
        unsafe_allow_html=True,
    )

    # Visualization Section
    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

    # Simplified Moving Average Graph (MA100)
    st.subheader('Price vs MA100')
    ma_100_days = data['Close'].rolling(100).mean()
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', color='green')
    plt.plot(ma_100_days, label='MA100', color='blue')
    plt.legend()
    plt.title('Close Price vs MA100')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig1)

else:
    st.markdown('<div class="warning-box">Not enough data available for prediction.</div>', unsafe_allow_html=True)

# Section: Compare Multiple Stocks
st.markdown('<div class="section-title">Compare Stocks</div>', unsafe_allow_html=True)
stocks_to_compare = st.text_input('Enter up to 3 Stock Symbols (comma-separated)', 'AAPL, MSFT, TSLA')

comparison_data = []

if stocks_to_compare:
    stock_list = [s.strip() for s in stocks_to_compare.split(',')[:3]]
    for stock in stock_list:
        data = fetch_data(stock, start='2020-01-01', end=end)
        if not data.empty and len(data) > 1:
            comparison_data.append({
                'Stock': stock,
                'Latest Price': data['Close'].iloc[-1],
                'Price 6 Months Ago': data['Close'].iloc[0],
                '6-Month Growth (%)': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)

    # Ensure '6-Month Growth (%)' column is numeric
    comparison_df['6-Month Growth (%)'] = pd.to_numeric(comparison_df['6-Month Growth (%)'], errors='coerce')

    # Drop NaN values in '6-Month Growth (%)'
    comparison_df.dropna(subset=['6-Month Growth (%)'], inplace=True)

    # Reset index for proper alignment in .highlight_max()
    comparison_df = comparison_df.reset_index(drop=True)

    st.subheader('Stock Comparison Table')
    st.dataframe(comparison_df.style.highlight_max(subset=['6-Month Growth (%)'], color='green', axis=0))
else:
    st.markdown('<div class="warning-box">No valid data available for comparison.</div>', unsafe_allow_html=True)

# Define the date range
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# List of Indian stock tickers
all_indian_stocks = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
    'ADANIGREEN.NS', 'DMART.NS', 'WIPRO.NS', 'TECHM.NS', 'LT.NS',
    'ULTRACEMCO.NS', 'MARUTI.NS', 'BAJAJFINSV.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'
]

# Fetch historical data
data = fetch_data(all_indian_stocks, start=start_date, end=end_date)

# Handle missing data
if 'Adj Close' in data:
    adj_close_data = data['Adj Close']
elif 'Close' in data:
    adj_close_data = data['Close']
else:
    adj_close_data = pd.DataFrame()

if not adj_close_data.empty and len(adj_close_data) > 1:
    returns = (adj_close_data.iloc[-1] / adj_close_data.iloc[0] - 1) * 100
    top_performers = returns.sort_values(ascending=False).head(10)
    top_performers_df = pd.DataFrame(top_performers, columns=['6M Return (%)'])

    st.dataframe(top_performers_df)

else:
    st.markdown('<div class="warning-box">No data available for Indian stocks. Please try again later.</div>', unsafe_allow_html=True)

# Additional Recommendations
st.markdown("""
### Investment Tips:
1. **Diversify Your Portfolio**
2. **Do Your Research**
3. **Consult a Financial Advisor**
4. **Consider Risk Tolerance**
""")
