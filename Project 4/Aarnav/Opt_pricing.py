import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import scipy.stats as si
import numpy as np

def d1(S, K, T, r, sigma):
    D1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return D1

def d2(S, K, T, r, sigma):
    dd1 = d1(S, K, T, r, sigma)
    D2 = dd1 - sigma * np.sqrt(T)
    return D2

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if option_type == 'call':
        option_price = (S * si.norm.cdf(d1(S, K, T, r, sigma), 0.0, 1.0)) - (K * np.exp(-r * T) * si.norm.cdf(d2(S, K, T, r, sigma), 0.0, 1.0))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2(S, K, T, r, sigma), 0.0, 1.0)) - (S * si.norm.cdf(-d1(S, K, T, r, sigma), 0.0, 1.0))
    return option_price

def calc_delta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return si.norm.cdf(D1)
    elif option_type == 'put':
        return si.norm.cdf(D1) - 1

def calc_gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return si.norm.pdf(D1) / (S * sigma * np.sqrt(T))

def calc_theta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    term1 = -(S * si.norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * si.norm.cdf(D2)
        return term1 - term2
    elif option_type == 'put':
        term2 = r * K * np.exp(-r * T) * si.norm.cdf(-D2)
        return term1 + term2

def calc_vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * si.norm.pdf(D1) * np.sqrt(T)


def stock_graph(sym):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sym.index, y=sym['Close'].values.flatten()))
    st.plotly_chart(fig)

start_date = "2011-01-01"
end_date = "2024-01-01"

st.title("OPTION PRICING MODEL")
ticker=st.text_input("Enter ticker symbol")
if ticker:
    tables=yf.download(ticker,start=start_date,end=end_date)
    with st.sidebar:
        st.title("Graph of Stock price")
        stock_graph(tables)

    stock = yf.Ticker(ticker)
    S = stock.info['currentPrice']
    st.write(f"The current price is {S}")
    K = st.slider('Strike Price', min_value=0, max_value=int(S*2), value=int(S))
    T = st.slider("Time (in years)", min_value=0.0, max_value=10.0, value=1.0)
    r = st.slider("Risk-Free Rate (as a percentage)", min_value=0.0, max_value=10.0, value=5.0) / 100
    sigma = st.slider("Volatility (as a percentage)", min_value=0.0, max_value=100.0, value=20.0) / 100

    call_button = st.button("Call Option Price")
    put_button = st.button("Put Option Price")

    if call_button:
            option_price = black_scholes(S, K, T, r, sigma, option_type='call')
            st.write(f"The Call option price is: {option_price}")
            st.write(f"DELTA:   {calc_delta(S, K, T, r, sigma, option_type='call')}")
            st.write(f"GAMMA:   {calc_gamma(S, K, T, r, sigma)}")
            st.write(f"THETA:   {calc_theta(S, K, T, r, sigma, option_type='call')}")
            st.write(f"VEGA:   {calc_vega(S, K, T, r, sigma)}")
    elif put_button:
            option_price = black_scholes(S, K, T, r, sigma, option_type='put')
            st.write(f"The Put option price is: {option_price}")
            st.write(f"DELTA:   {calc_delta(S, K, T, r, sigma, option_type='put')}")
            st.write(f"GAMMA:   {calc_gamma(S, K, T, r, sigma)}")
            st.write(f"THETA:   {calc_theta(S, K, T, r, sigma, option_type='put')}")
            st.write(f"VEGA:   {calc_vega(S, K, T, r, sigma)}")




