import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
from math import exp, sqrt, log
from scipy.stats import norm

# Black-Scholes formula for Call and Put option pricing with check for T=0
def black_scholes(S, K, T, r, sigma, option_type="call"):
    # Set a small value for T if it's zero to avoid division by zero
    T = max(T, 1e-6)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Option Greeks for both Call and Put options
def option_greeks(S, K, T, r, sigma):
    T = max(T, 1e-6)  # Ensure time to expiration is non-zero
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))) - r * K * exp(-r * T) * norm.cdf(d2)
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T))) + r * K * exp(-r * T) * norm.cdf(-d2)
    vega = S * norm.pdf(d1) * sqrt(T)
    rho_call = K * T * exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * exp(-r * T) * norm.cdf(-d2)
    
    return delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put

# Streamlit UI setup
st.title("Option Pricing & Greeks Calculator")
st.header("**Created by Kalash Jain 📈📊**")
st.sidebar.header("User Input Parameters")

# User inputs for stock details
stock_price = st.sidebar.number_input("Current Stock Price", value=100.0, step=1.0)
strike_price = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (in %)", value=5.0) / 100
volatility = st.sidebar.number_input("Volatility (in %)", value=20.0) / 100
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")

# Input current and expiry month/year
current_year = st.sidebar.number_input("Current Year", value=datetime.today().year, step=1)
current_month = st.sidebar.number_input("Current Month", value=datetime.today().month, step=1, min_value=1, max_value=12)

expiry_year = st.sidebar.number_input("Expiry Year", value=datetime.today().year + 1, step=1)
expiry_month = st.sidebar.number_input("Expiry Month", value=datetime.today().month, step=1, min_value=1, max_value=12)

# Calculate time to expiration in years
months_to_expiration = (expiry_year - current_year) * 12 + (expiry_month - current_month)
time_to_expiration = months_to_expiration / 12.0

# Check if the expiry date is before the current date
if time_to_expiration <= 0:
    st.sidebar.warning("Expiry date must be later than current date. Using a small positive value for time to expiration.")
    time_to_expiration = 1e-6  # Small value to prevent division by zero

# Display calculated time to expiration
st.sidebar.write(f"Time to Expiration (in years): {time_to_expiration:.6f}")

# Calculate Call and Put option prices
call_price = black_scholes(stock_price, strike_price, time_to_expiration, risk_free_rate, volatility, "call")
put_price = black_scholes(stock_price, strike_price, time_to_expiration, risk_free_rate, volatility, "put")

# Display option prices
st.write(f"### Call Option Price: ${call_price:.2f}")
st.write(f"### Put Option Price: ${put_price:.2f}")

# Calculate and display Option Greeks for Call and Put options
greeks = option_greeks(stock_price, strike_price, time_to_expiration, risk_free_rate, volatility)
greek_labels = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']

# Create a DataFrame-like structure for separate columns
greeks_data = {
    "Greek": greek_labels,
    "Call Option": [greeks[0], greeks[2], greeks[3], greeks[5], greeks[6]],
    "Put Option": [greeks[1], greeks[2], greeks[4], greeks[5], greeks[7]]
}

# Display the Option Greeks in a column-separated table
st.write("### Option Greeks for Call and Put Options")
st.table(greeks_data)

# Graphs for Stock Price
st.write(f"### Stock Price Graph for {ticker}")

# Fetch stock data using yfinance
stock_data = yf.Ticker(ticker)
stock_history = stock_data.history(period='1y')

# Line chart for stock price
st.line_chart(stock_history['Close'], width=700, height=400)

# Candlestick chart for stock price
st.write("### Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(x=stock_history.index,
                open=stock_history['Open'],
                high=stock_history['High'],
                low=stock_history['Low'],
                close=stock_history['Close'])])
st.plotly_chart(fig)

# Show data table
st.write("### Stock Data Table")
st.dataframe(stock_history)