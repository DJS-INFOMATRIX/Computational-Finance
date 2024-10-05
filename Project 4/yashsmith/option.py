import streamlit as st
import numpy as np
import plotly.graph_objs as go
from math import exp, sqrt, log
from scipy.stats import norm
import pandas as pd

# Set page config
st.set_page_config(page_title="Options Analytics", layout="wide")

# Improved CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stHeader {
        background-color: #262730;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stSubheader {
        color: #4fbdff;
        margin-top: 2rem;
        font-size: 1.5em;
    }
    div[data-testid="stMetricValue"] {
        background-color: #262730;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
        color: #4fbdff;
    }
    div.stButton > button {
        background-color: #4fbdff;
        color: #0e1117;
    }
    div[data-testid="stDataFrameResizable"] {
        background-color: #262730;
        border: 1px solid #4fbdff;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .dataframe {
        font-size: 1.1em;
    }
    .dataframe th {
        background-color: #4fbdff;
        color: #0e1117;
        padding: 0.5em;
    }
    .dataframe td {
        background-color: #1e2130;
        color: #ffffff;
        padding: 0.5em;
    }
    .stNumberInput > div > div > input {
        color: #ffffff;
    }
    .stSelectbox > div > div > select {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    
    if option_type == "call":
        return S * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2)
    elif option_type == "put":
        return K * exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

def calculate_greeks(S, K, T, r, sigma):
    T = max(T, 1e-6)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    
    delta_call = norm.cdf(d_1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d_1) / (S * sigma * sqrt(T))
    vega = S * sqrt(T) * norm.pdf(d_1) / 100
    theta_call = (-S * norm.pdf(d_1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d_2)) / 365
    theta_put = (-S * norm.pdf(d_1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d_2)) / 365
    rho_call = K * T * exp(-r * T) * norm.cdf(d_2) / 100
    rho_put = -K * T * exp(-r * T) * norm.cdf(-d_2) / 100
    
    return {
        'Delta': {'Call': delta_call, 'Put': delta_put},
        'Gamma': {'Call': gamma, 'Put': gamma},
        'Vega': {'Call': vega, 'Put': vega},
        'Theta': {'Call': theta_call, 'Put': theta_put},
        'Rho': {'Call': rho_call, 'Put': rho_put}
    }

def main():
    st.markdown('<div class="stHeader"><h1>🎯 Options Analytics Platform</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="stSubheader">Input Parameters</h2>', unsafe_allow_html=True)
        S = st.number_input("Stock Price ($)", value=100.0, step=0.01)
        K = st.number_input("Strike Price ($)", value=100.0, step=0.01)
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
        sigma = st.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
        
        expiry_options = {
            '1 Week': 1/52,
            '2 Weeks': 2/52,
            '1 Month': 1/12,
            '2 Months': 2/12,
            '3 Months': 0.25,
            '6 Months': 0.5,
            '9 Months': 0.75,
            '1 Year': 1,
            '18 Months': 1.5,
            '2 Years': 2,
            '3 Years': 3,
            '5 Years': 5
        }
        T = st.selectbox("Time to Expiration", list(expiry_options.keys()))
        T_value = expiry_options[T]
    
    with col2:
        st.markdown('<h2 class="stSubheader">Options Pricing Analysis</h2>', unsafe_allow_html=True)
        
        call_price = black_scholes(S, K, T_value, r, sigma, "call")
        put_price = black_scholes(S, K, T_value, r, sigma, "put")
        
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            st.metric("Call Option Price", f"${call_price:.2f}")
        with price_col2:
            st.metric("Put Option Price", f"${put_price:.2f}")
        
        greeks = calculate_greeks(S, K, T_value, r, sigma)
        
        st.markdown('<h2 class="stSubheader">Greeks Analysis</h2>', unsafe_allow_html=True)
        greek_df = pd.DataFrame(greeks).round(4)
        st.dataframe(greek_df, use_container_width=True, height=150)  # Increased height
        
        # Sensitivity Analysis
        st.markdown('<h2 class="stSubheader">Price Sensitivity Analysis</h2>', unsafe_allow_html=True)
        price_range = np.linspace(S * 0.7, S * 1.3, 100)
        calls = [black_scholes(p, K, T_value, r, sigma, "call") for p in price_range]
        puts = [black_scholes(p, K, T_value, r, sigma, "put") for p in price_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=calls, name="Call Option", line=dict(color='#4fbdff')))
        fig.add_trace(go.Scatter(x=price_range, y=puts, name="Put Option", line=dict(color='#ffa14f')))
        fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=max(max(calls), max(puts)),
                      line=dict(color="white", width=2, dash="dash"))
        fig.update_layout(
            title="Put option vs Call option",
            xaxis_title="Stock Price ($)",
            yaxis_title="Option Price ($)",
            height=400,
            plot_bgcolor='rgba(30,33,48,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()