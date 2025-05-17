
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

# Initialize Ollama model (make sure you have ollama running and model pulled)
chat = ChatOllama(model="mistral", temperature=0.5)
# Set up chat history in session state
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a comedian AI assistant")
    ]

# Function to get response from the Ollama model
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# Input box
input = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

# If the submit button is clicked
if submit and input:
    response = get_chatmodel_response(input)
    st.subheader("The Response is")
    st.write(response)
