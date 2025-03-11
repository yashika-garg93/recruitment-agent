import streamlit as st
import openai
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ---------- SETUP ---------- #
st.set_page_config(page_title="AI Recruitment Agent", layout="wide")
st.title("🤖 AI Recruitment Agent (RAG-powered)")
st.markdown("Upload your background once. Provide JD links and application questions. Get personalized responses instantly!")

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.warning("No OpenAI API key found! Add it in Streamlit Secrets.")

# Initialize Embeddings & ChromaDB
embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
chroma_client = Chroma(collection_name="user_background", embedding_function=embedding_model)

# ---------- PDF INGESTION FUNCTION ---------- #
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ---------- STEP 1: Upload & Embed Background Files ---------- #
st.header("Step 1: Upload Your Background Info (Resume, Projects, etc.)")

uploaded_files = st.file_uploader("Upload multiple PDF files (Only once)", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    st.info(f"{len(uploaded_files)} files uploaded. Extracting and storing...")
    documents = []
    for file in uploaded_files:
        file_text = extract_text_from_pdf(file)
        documents.append(file_text)
    
    # Store embeddings in ChromaDB
    chroma_client.add_texts(documents)
    st.success("✅ Background information stored in memory!")

# ---------- STEP 2: Job Description Scraper ---------- #
st.header("Step 2: Provide the Job Description Link")

jd_url = st.text_input("Paste the Job Description URL (optional)")

def scrape_jd(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        return f"Error scraping JD: {e}"

jd_text = ""

if jd_url:
    jd_text = scrape_jd(jd_url)
    st.success("✅ JD scraped successfully!")
    st.text_area("Job Description", jd_text, height=200)

# ---------- STEP 3: Application Questions Input ---------- #
st.header("Step 3: Enter the Application Questions")

questions = st.text_area("Paste your application questions here (one per line):")

questions_list = []
if questions:
    questions_list = [q.strip() for q in questions.split('\n') if q.strip()]

# ---------- STEP 4: Retrieve Background Info for Each Question ---------- #
def retrieve_relevant_background(question):
    retriever = chroma_client.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join([doc.page_content for doc in docs])

# ---------- STEP 5: Generate Responses with RAG + GPT-4 ---------- #
st.header("Step 4: Generate Answers")

if st.button("Generate Answers"):
    if not jd_text:
        st.warning("Please provide a Job Description link first!")
    elif not questions_list:
        st.warning("Please enter application questions.")
    else:
        st.info("Generating responses...")

        for question in questions_list:
            st.markdown(f"### ✏️ {question}")
            retrieved_background = retrieve_relevant_background(question)

            prompt = f"""
            You are an AI assistant helping a candidate apply for a job.

            Candidate Background (retrieved from their documents):
            {retrieved_background}

            Job Description:
            {jd_text}

            Application Question:
            {question}

            Generate a professional, detailed, personalized response.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": "You are a job application assistant."},
                          {"role": "user", "content": prompt}]
            )

            answer = response["choices"][0]["message"]["content"]
            st.text_area(f"Generated Answer for: {question}", answer, height=200)

# ---------- End ---------- #
st.markdown("---")
st.caption("Built with ❤️ by Yashika's AI Recruitment Agent (RAG-Powered)")
