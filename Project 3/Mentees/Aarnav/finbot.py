import streamlit as st
import speech_recognition as sr
from huggingface_hub import InferenceClient
import os
from sklearn.linear_model import LinearRegression
import tempfile
import plotly.graph_objects as go
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from gtts import gTTS


load_dotenv()

def generate_ai_response(prompt, client):
    csv_data= st.session_state.get('csv_data', pd.DataFrame())
    text_data= st.session_state.get('text_data', "No text data available")
    system_message = f"""You are a personal finance AI assistant. You will only respond to queries related to personal finance.
    Also answer questions based on the following data:

    Text data:
    {text_data}

    CSV data:
    {csv_data.to_string() if not csv_data.empty else "No CSV data available"}
    When answering questions, always refer to and use the information from the Text data and CSV data provided above if it's relevant to the query.
    Answer questions related to the uploaded files or personal finance only."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=120,
        stream=True
    ):
        response += message.choices[0].delta.content or ""

    return response

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {str(e)}")
        return None

def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Speak now.")
            audio = recognizer.listen(source)
        st.write("Processing speech...")
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
    except Exception as e:
        return f"An error occurred: {str(e)}"
            
def stock_graph(sym):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sym.index, y=sym['Close'].values.flatten()))
    st.plotly_chart(fig)

def predict(ticker):
    start_date = "2011-01-01"
    end_date = "2024-01-01"
    tbl=yf.download(ticker,start=start_date,end=end_date)
    tbl['Average']=tbl['High']/2 +tbl['Low']/2
    X = tbl['Close'].values.reshape(-1,1)
    y = tbl[['Open','Average']].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred=model.predict(X)
    tbl['Pred_Open']=y_pred[:,0]
    tbl['Pred_Average']=y_pred[:,1]
    tbl['Old_close']=tbl['Close'].shift(1)
    tbl['Old_open']=tbl['Pred_Open'].shift(1)
    tbl['Old_Average']=tbl['Pred_Average'].shift(1)
    tbl = tbl.iloc[1:]
    X = tbl[['Old_open','Old_close','Old_Average']].values
    y = tbl['Close'].values
    model1 = LinearRegression()
    model1.fit(X, y)
    stock_info = yf.Ticker(ticker).info
    cls_price=stock_info.get('currentPrice')
    y_prd=model.predict([[float(cls_price)]])
    a=y_prd[0][0]
    b=y_prd[0][1]
    y_fnl=model1.predict([[a,float(cls_price),b]])
    return y_fnl[0]
    

def main():
    st.title('AI Assistant')

    
    with st.sidebar:
        st.title('AI Assistant Settings')
        hf_api_token = os.getenv("Hugging_Face_Api")
        st.button('Clear Chat History', on_click=clear_chat_history)
        st.title("Drag and Drop Files")
        csv_uploaded_files = st.file_uploader("Drag and drop CSV files here", type=["csv"])
        if csv_uploaded_files is not None:
            st.session_state.csv_data = pd.read_csv(csv_uploaded_files)
            st.success("CSV data uploaded and ready to use!")

        text_uploaded_files = st.file_uploader("Drag and drop text files here", type=["txt"])
        if text_uploaded_files is not None:
            text_data = text_uploaded_files.getvalue().decode("utf-8")
            st.session_state.text_data = text_data
            st.success("Text data uploaded and ready to use!")

        st.title("Stock Price Predictor")
        tick=st.chat_input("Enter Ticker Symbol")
        if tick:
            st.write("Tomorrow's Predicted Stock price is ",predict(tick))
        else:
            st.write("Enter Valid ticker symbol")

    
    client = InferenceClient(
        token=hf_api_token  
    )

    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    user_input = st.chat_input("Type your message here:")
    response = ""

    if user_input:
        start_date = "2011-01-01"
        end_date = "2024-01-01"
        if "stock price of" in user_input.lower():
            ticker = user_input.lower().split("stock price of")[-1].strip()
            tables=yf.download(ticker,start=start_date,end=end_date)
            try:
                stock_info = yf.Ticker(ticker).info
                current_price = stock_info.get('currentPrice')
                response = f"\nThe current price of {ticker} is: {current_price}"
            except Exception as e:
                response = f"\nError fetching stock data: {e}"
            with st.sidebar:
                st.title("Graph of Stock price")
                stock_graph(tables)
                
        elif "stock price table of" in user_input.lower():
            ticker = user_input.lower().split("stock price table of")[-1].strip()
            tables=yf.download(ticker,start=start_date,end=end_date)
            response=st.dataframe(tables)
            with st.sidebar:
                st.title("Graph of Stock price")
                stock_graph(tables)
                


        else:
            response = generate_ai_response(user_input, client)

        
        if response:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    
    if st.button("Read", key=f"tts_{message}"):
        audio_file = text_to_speech(message["content"])
        if audio_file:
            st.audio(audio_file)

    if st.button("Speak"):
        speech = recognize_speech()
        if speech:
            user_input=speech
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = generate_ai_response(user_input, client)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        

            
if __name__ == "__main__":
    main()
