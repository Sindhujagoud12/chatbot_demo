import streamlit as st
from chatbot import Chatbot
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI key (if available)
openai_key = os.getenv("OPENAI_API_KEY")

# Sidebar
st.sidebar.title("💬 Chatbot Settings")
backend = st.sidebar.radio("Choose chatbot mode:", ["local", "openai"])

# Initialize chatbot
if backend == "openai":
    if not openai_key:
        st.error("⚠️ Please set your OPENAI_API_KEY in the .env file.")
    cb = Chatbot(backend="openai", openai_api_key=openai_key)
else:
    cb = Chatbot(backend="local")

# App title
st.title("🤖 Simple Chatbot Demo")

# Chat input
user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send") and user_input:
    reply = cb.get_reply(user_input)
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(reply)
