import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_ai_response(prompt, client):
    system_message = "You are a helpful AI assistant. Provide concise responses."
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

def main():
    st.title('AI Assistant')

    # Sidebar for settings
    with st.sidebar:
        st.title('🤖 AI Assistant Settings')
        hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        st.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize the InferenceClient
    client = InferenceClient(
        "{insert_here}",
        token=hf_api_token
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your message here:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        response = generate_ai_response(user_input, client)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()