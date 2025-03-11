import streamlit as st
import openai
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ---------- STREAMLIT CONFIG ---------- #
st.set_page_config(page_title="AI Recruitment Agent", layout="wide")

st.title("🤖 AI Recruitment Agent (RAG-powered)")
st.markdown("""
Upload your background once. Provide JD links and application questions.  
Get personalized responses instantly!
""")

# ---------- USER ENTERS THEIR OPENAI API KEY ---------- #
st.sidebar.header("🔑 API Key Setup")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.sidebar.warning("⚠️ Please enter your OpenAI API Key to proceed.")
    st.stop()

# ---------- INITIALIZE OPENAI + CHROMA MEMORY ---------- #
openai.api_key = openai_api_key

# Initialize OpenAI Embeddings (RAG)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
chroma_client = Chroma(collection_name="user_background", embedding_function=embedding_model)

# ---------- UTILITY FUNCTION: EXTRACT TEXT FROM PDF ---------- #
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ---------- STEP 1: UPLOAD USER BACKGROUND FILES ---------- #
st.header("Step 1: Upload Your Background 
