import streamlit as st 
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from gtts import  gTTS
import tempfile
import speech_recognition as sr
import plotly.graph_objs as go
import re
from collections import Counter

# Load environment variables
load_dotenv()

# Function to load fine-tuning data
def load_fine_tuning_data():
    fine_tuning_file = 'tune_data.txt'
    if os.path.exists(fine_tuning_file):
        with open(fine_tuning_file, 'r') as file:
            return file.read()
    return ""

# Function to load CSV data
def load_csv_data():
    csv_file = 'bot_score.csv'
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()

# Load fine-tuning data and CSV data
fine_tuning_data = load_fine_tuning_data()
csv_data = load_csv_data()

def fetch_stock_data(ticker):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        company_name = info.get('longName', 'Unknown Company')
        return f"{company_name} (${ticker}) current price: ${current_price:.2f}"
    except Exception as e:
        return f"Unable to fetch information for {ticker}. Error: {str(e)}"

def fetch_stock_history(ticker, period='1mo'):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    fig = go.Figure(data=go.Scatter(x=history.index, y=history['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def generate_ai_response(prompt, client):

    # Add sentiment analysis
    sentiment = analyze_sentiment(prompt)
    
    # Add keyword extraction
    keywords = extract_keywords(prompt)
    
    system_message =f"""You are a helpful Fintech AI assistant who answers only queries related to finance,including topics like 
    the stock market, mutual funds, investment banking,retirement planning,financial planning,, market trends, 
    personal finance management, and financial products and services.Please refrain from providing information or engaging in 
    discussions that do not pertain to these finance topics.
    Use the following additional information in your responses:
    
    Text data:
    {fine_tuning_data}

    CSV data:
    {csv_data.to_string() if not csv_data.empty else "No CSV data available"}

    For stock queries, use the provided stock information and suggest checking the stock chart for historical performance.
   
    User's message sentiment: {sentiment}
    Key topics in user's message: {', '.join(keywords)}
    
    Tailor your response based on the sentiment and keywords. If the sentiment is negative, be more empathetic in your response. Focus on addressing the key topics identified."""

    if "stock" in prompt.lower() or "$" in prompt:
        words = prompt.replace("$", "").split()
        potential_tickers = [word.upper() for word in words if word.isalpha() and len(word) <= 5]
        
        #stock_info = ""
        for ticker in potential_tickers:
           # stock_info += fetch_stock_data(ticker) + "\n"
            try:
                fig = fetch_stock_history(ticker)
                st.plotly_chart(fig)
            except:
                st.write(f"Couldn't fetch stock data for {ticker}")
        
        #if stock_info:
            #prompt += f"\n\nHere's the current stock information:\n{stock_info}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response = ""
    try:
        for message in client.chat_completion(
            messages=messages,
            max_tokens=150,
            stream=True
        ):
            response += message.choices[0].delta.content or ""
    except Exception as e:
        if "429" in str(e):
            st.error("We've hit a rate limit. Please try again in a few moments.")
        else:
            st.error(f"An error occurred: {e}")
        response = "Sorry, there was an error processing your request."
   
    return response, sentiment, keywords

def analyze_sentiment(text):
    # This is a simple rule-based sentiment analysis. You could use a more sophisticated model.
    positive_words = set(['good', 'great', 'excellent', 'happy', 'positive'])
    negative_words = set(['bad', 'poor', 'terrible', 'unhappy', 'negative'])
    
    words = text.lower().split()
    sentiment_score = sum(word in positive_words for word in words) - sum(word in negative_words for word in words)
    
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(text):
    # This is a simple keyword extraction. You could use more advanced NLP techniques.
    stopwords = set(['the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'is', 'are'])
    words = text.lower().split()
    return [word for word in words if word not in stopwords and len(word) > 3]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

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
    
def summarize_conversation(messages):
    summary = "Conversation Summary:\n\n"
    for msg in messages:
        if msg["role"] == "user":
            summary += f"User: {msg['content'][:50]}...\n"
        else:
            summary += f"Assistant: {msg['content'][:50]}...\n"
        summary += f"Sentiment: {msg.get('sentiment', 'N/A')}\n"
        summary += f"Keywords: {', '.join(msg.get('keywords', []))}\n\n"
    return summary    


def main():
    st.title('AI Assistant')

    # Sidebar for settings
    with st.sidebar:
        st.title('🤖 AI Assistant Settings')
        hf_api_token = os.getenv(HUGGING_FACE_TOKEN)
        st.button('Clear Chat History', on_click=clear_chat_history)

        uploaded_txt_file = st.file_uploader("Upload fine-tuning data (TXT)", type="txt")
        if uploaded_txt_file is not None:
            fine_tuning_data = uploaded_txt_file.getvalue().decode("utf-8")
            with open('fine_tuning_data.txt', 'w') as f:
                f.write(fine_tuning_data)
            st.success("Fine-tuning text data uploaded and saved!")

        uploaded_csv_file = st.file_uploader("Upload CSV data", type="csv")
        if uploaded_csv_file is not None:
            csv_data = pd.read_csv(uploaded_csv_file)
            csv_data.to_csv('data.csv', index=False)
            st.success("CSV data uploaded and saved!")  

        if st.button("Summarize Conversation"):
            summary = summarize_conversation(st.session_state.messages)
            st.text_area("Conversation Summary", summary, height=300)

    # Initialize the InferenceClient
    client = InferenceClient(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=HUGGING_FACE_TOKEN
     )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Create a placeholder for chat messages
    chat_placeholder = st.empty()

    # Create a container for input at the bottom
    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            user_input = st.chat_input("Ask about finances, stocks, or insurance:")
        with col2:
            speak_button = st.button("🎤")

    # Handle the speak button
    if speak_button:
        user_input = speech_to_text()
        st.write(f"You said: {user_input}")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        response, sentiment, keywords = generate_ai_response(user_input, client)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sentiment": sentiment,
            "keywords": keywords})

    # Display chat messages in the placeholder
    with chat_placeholder.container():
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    st.caption(f"Detected sentiment: {message.get('sentiment', 'N/A')}")
                    st.caption(f"Key topics: {', '.join(message.get('keywords', []))}")
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    if st.button("🔊", key=f"play_{i}"):
                        audio_file = text_to_speech(message["content"])
                        st.audio(audio_file)

if __name__ == "__main__":
    main()