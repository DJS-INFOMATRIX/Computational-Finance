import streamlit as st
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from gtts import gTTS
import tempfile
import speech_recognition as sr
import replicate
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import random

# Load environment variables
load_dotenv()

# Set the API token as an environment variable
os.environ["REPLICATE_API_TOKEN"] = "r8_90fjjALmLHP2Gfc9Tc5axQYtw9tngTI1u0Ge4"

def load_fine_tuning_data():
    fine_tuning_file = 'tune_data.txt'
    if os.path.exists(fine_tuning_file):
        with open(fine_tuning_file, 'r') as file:
            return file.read()
    return ""

def load_csv_data():
    csv_file = 'data.csv'
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()

fine_tuning_data = load_fine_tuning_data()
csv_data = load_csv_data()

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        company_name = info.get('longName', 'Unknown Company')
        return f"{company_name} (${ticker}) current price: ${current_price:.2f}"
    except Exception as e:
        return f"Unable to fetch information for {ticker}. Error: {str(e)}"

def generate_stock_chart(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
    
    fig.update_layout(title=f'{ticker} Stock Price - Past Year',
                      xaxis_title='Date',
                      yaxis_title='Price')
    
    return fig

def get_news(query):
    api_key = "55f8b079335a4cdda914e2a67621de7a"  # Replace with your actual News API key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
    
    response = requests.get(url)
    if response.status_code == 200:
        news = response.json()
        return news['articles']
    else:
        return None

def is_finance_related(prompt):
    finance_keywords = ["finance", "stock", "investment", "money", "bank", "credit", "loan", "insurance", "budget", "saving", "debt", "economy", "market", "fund", "portfolio", "dividend", "asset", "liability", "interest", "mortgage", "tax", "retirement", "401k", "IRA"]
    return any(keyword in prompt.lower() for keyword in finance_keywords)

def generate_ai_response(prompt):
    if not is_finance_related(prompt):
        return "I'm sorry, but I can only answer questions related to finance, stocks, investments, and other financial matters. Could you please ask a finance-related question?"

    system_message = f"""You are an AI Assistant specialized in personal finance, insurance, credit scoring, stocks, and related topics. 
    Only answer questions related to finance. If a question is not about finance, politely inform the user that you can only discuss financial matters.
    Use the following additional information in your responses:

    Text data:
    {fine_tuning_data}

    CSV data:
    {csv_data.to_string() if not csv_data.empty else "No CSV data available"}

    For stock queries, use the provided stock information."""

    if "stock" in prompt.lower() or "$" in prompt:
        words = prompt.replace("$", "").split()
        potential_tickers = [word.upper() for word in words if word.isalpha() and len(word) <= 5]
        
        stock_info = ""
        for ticker in potential_tickers:
            stock_info += get_stock_info(ticker) + "\n"
        
        if stock_info:
            prompt += f"\n\nHere's the current stock information:\n{stock_info}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    try:
        output = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e",
            input={
                "prompt": str(messages),
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        response = ''.join(output)
    except Exception as e:
        st.error(f"An error occurred during AI response generation: {str(e)}")
        response = "Sorry, there was an error processing your request. Please check your API token and try again."

    return response

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Financial Assistant. How may I help you with financial matters today?"}]

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now.")
        audio = r.listen(source)
        st.write("Processing speech...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error processing your speech."

def set_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            color: #FFFFFF;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            color: #FFFFFF;
            background-color: #2E2E2E;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stButton>button {
            color: #000000;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            color: #000000;
            background-color: #F0F0F0;
        }
        .stMarkdown {
            color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)

def stock_prediction_game():
    game_content = st.empty()
    
    with game_content.container():
        st.subheader("Stock Price Prediction Game")
        
        if 'game_score' not in st.session_state:
            st.session_state.game_score = 0
        
        ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", key="game_ticker")
        if ticker:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            st.write(f"Current price of {ticker}: ${current_price:.2f}")
            
            prediction = st.radio("Do you think the stock price will go up or down tomorrow?", ('Up', 'Down'))
            if st.button("Submit Prediction"):
                # Simulate next day's price (random for demonstration)
                next_day_price = current_price * (1 + random.uniform(-0.05, 0.05))
                st.write(f"Next day's price: ${next_day_price:.2f}")
                
                if (prediction == 'Up' and next_day_price > current_price) or (prediction == 'Down' and next_day_price < current_price):
                    st.success("Correct prediction! You earned a point.")
                    st.session_state.game_score += 1
                else:
                    st.error("Wrong prediction. Better luck next time!")
                
                st.write(f"Your current score: {st.session_state.game_score}")

def main():
    st.markdown(
    """
    <style>
    @font-face {
        font-family: 'TattooFont';
        src: url('path_to_your_font_file.ttf') format('truetype');
    }
    .tattoo-text {
        font-family: 'TattooFont';
        color: gold;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("<h1 class='tattoo-text'>AI Financial Assistant</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.title('🤖 AI Assistant Settings')
        
        theme = st.radio("Choose Theme", ("Light", "Dark"))
        set_theme(theme.lower())
        
        st.button('Clear Chat History', on_click=clear_chat_history)
        
        uploaded_txt_file = st.file_uploader("Upload fine-tuning data (TXT)", type="txt")
        if uploaded_txt_file is not None:
            fine_tuning_data = uploaded_txt_file.getvalue().decode("utf-8")
            with open('tune_data.txt', 'w') as f:
                f.write(fine_tuning_data)
            st.success("Fine-tuning text data uploaded and saved!")

        uploaded_csv_file = st.file_uploader("Upload CSV data", type="csv")
        if uploaded_csv_file is not None:
            csv_data = pd.read_csv(uploaded_csv_file)
            csv_data.to_csv('data.csv', index=False)
            st.success("CSV data uploaded and saved!")

        st.subheader("Stock Visualization")
        stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", key="sidebar_ticker")
        if stock_ticker:
            st.plotly_chart(generate_stock_chart(stock_ticker))
        
        # Add the stock prediction game to the sidebar
        if 'show_game' not in st.session_state:
            st.session_state.show_game = False
        
        if st.button("Toggle Stock Prediction Game"):
            st.session_state.show_game = not st.session_state.show_game

    # Display the game in the main content area if toggled on
    if st.session_state.show_game:
        stock_prediction_game()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Financial Assistant. How may I help you with financial matters today?"}]

    chat_placeholder = st.empty()

    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            user_input = st.chat_input("Ask about finances, stocks, or insurance:")
        with col2:
            speak_button = st.button("🎤")

    if speak_button:
        user_input = speech_to_text()
        st.write(f"You said: {user_input}")

    if user_input:
        if is_finance_related(user_input):
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            response = generate_ai_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display relevant news
            news = get_news(user_input)
            if news:
                st.subheader("Related News")
                for article in news:
                    st.write(f"**{article['title']}**")
                    st.write(article['description'])
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")
        else:
            st.warning("I'm sorry, but I can only answer questions related to finance. Could you please ask a finance-related question?")

    with chat_placeholder.container():
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    if st.button("🔊", key=f"play_{i}"):
                        audio_file = text_to_speech(message["content"])
                        st.audio(audio_file)

if __name__ == "__main__":
    main()