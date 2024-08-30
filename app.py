import os
import random
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.text_loaders import TextLoader
from langchain_groq import ChatGroq
import language_tool_python
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define the prompt template for generating questions
prompt = ChatPromptTemplate.from_template(
    """You are an IELTS tutor tasked with generating a writing task based on the context provided below.
    DO NOT include any model answers, explanations, or guidance on how to answer the question.
    Only generate the task prompt that follows IELTS standards for Task 2.
    
    <context>
    {context}
    </context>
    
    Task: Generate an IELTS Writing Task 2 prompt using the context above."""
)

# Define functions
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFLoader("pdf_files")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def get_random_question():
    """Get a random question from a predefined list."""
    questions = [
        "Describe a time when you had to manage a difficult situation.",
        "Discuss the impact of technology on education.",
        "Explain the advantages and disadvantages of living in a city.",
        "Describe a memorable event in your life."
    ]
    return random.choice(questions)

def get_feedback(text):
    """Provides feedback on the given text using LanguageTool."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    feedback = "\n".join([f"Error: {match.message} (at {match.offset})" for match in matches])
    return feedback

# Streamlit interface
st.set_page_config(page_title="IELTS Practice", layout="wide")

# Sidebar with a heading "Features"
st.sidebar.title("Features")

# Main options in the sidebar
option = st.sidebar.radio("Select an option:", ["Practice Session", "Mock Tests"])

# Subsections under "Practice Session"
if option == "Practice Session":
    st.sidebar.subheader("Practice Sections")
    practice_option = st.sidebar.selectbox("Choose a section:", ["Reading", "Listening", "Writing", "Speaking"])

# Display content based on the selection
if option == "Practice Session":
    st.title(f"Practice Session: {practice_option}")
    st.write(f"Content for {practice_option} section will go here.")
    if practice_option == "Writing":
        st.write("Click the button below to receive a random writing task question.")
        if st.button("Get Random Writing Task Question"):
            question = get_random_question()
            st.write("**Your Writing Task Prompt:**")
            st.write(question)
            
            # Provide a text area for users to submit their essays
            essay_text = st.text_area("Submit your essay for feedback:")
            if st.button("Submit Essay for Feedback"):
                if essay_text:
                    feedback = get_feedback(essay_text)  # Provide feedback on the essay
                    st.write("**Feedback:**")
                    st.write(feedback)
                else:
                    st.write("Please enter your essay to get feedback.")

elif option == "Mock Tests":
    st.title("Mock Tests")
    st.write("Mock test content will go here.")

# Document embedding and retrieval
text_file_path = 'path/to/your/questions.txt'  # Update this to the location of your text file
text_data = load_text(text_file_path)

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

user_prompt = st.text_input("Enter your query from the pdf files")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get('context', [])):
            st.write(doc.page_content)
            st.write('---------------')

# Simple styling for a clean interface
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .reportview-container .markdown-text-container {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
