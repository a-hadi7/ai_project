import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.header("📚 Chat with your Textbook (ACSAI Project)")

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.title("Settings")
    pdf_docs = st.file_uploader("Upload your PDF Books and click 'Process'", accept_multiple_files=True)
    api_key = st.text_input("Enter Gemini API Key", type="password")

# --- CORE FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    # Using the new gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the textbook context." Don't make up an answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # UPDATED: Using the brand-new gemini-1.5-flash model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- MAIN APP LOGIC ---
user_question = st.text_input("Ask a question about your uploaded PDFs:")

if user_question:
    if not api_key:
        st.error("Please enter your API Key in the sidebar first!")
    else:
        with st.spinner("Searching the textbook and generating answer..."):
            # Using the new gemini-embedding-001 model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            
            docs = new_db.similarity_search(user_question)

            chain = get_conversational_chain(api_key)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            st.write("### AI Response:")
            st.write(response["output_text"])

if st.sidebar.button("Process PDFs"):
    if not api_key:
        st.error("Please enter your API Key first!")
    elif not pdf_docs:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Extracting text, creating embeddings, and building database..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, api_key)
            st.success("Processing complete! You can now ask questions about your documents.")