import os
import random
import streamlit as st
from langchain.chat_models import ChatOpenAI  # Correct import for OpenAI integration
from langchain.embeddings.openai import OpenAIEmbeddings  # Correct import for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Correct import for text splitting
from langchain.vectorstores import FAISS  # Correct import for FAISS
from langchain.document_loaders import PyPDFDirectoryLoader  # Correct import for loading PDFs
from langchain.chains import RetrievalQA
import openai
from langchain_groq import ChatGroq
import tempfile  # For handling temporary files
from dotenv import load_dotenv

# Load environment variables for OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Make sure to set the OpenAI API key globally
openai.api_key = OPENAI_API_KEY

# Function to load and parse the PDF for Speaking Part 2
def create_vector_embedding_for_speaking_part2():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("speaking2_pdf")  # Load the directory where the PDF is saved
    st.session_state.docs = st.session_state.loader.load()  # Document loading
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.session_state.speaking2_tasks = extract_speaking2_questions(st.session_state.final_documents)

# Function to extract questions from the document
def extract_speaking2_questions(documents):
    tasks = []
    for doc in documents:
        content = doc.page_content
        if not content:
            continue

        # Split the content into lines
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            # Check if the line starts with a number (e.g., "1. Describe an internet business...")
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)  # Split on the first period
                if len(parts) == 2:
                    question = parts[1].strip()  # Get the question text
                    tasks.append({"question": question})  # Append it to the tasks list
    return tasks

# Function to generate similar questions using RAG
def generate_similar_questions_using_rag():
    if "speaking2_tasks" in st.session_state and st.session_state.speaking2_tasks:
        tasks = st.session_state.speaking2_tasks
        # Pick a random task
        random_task = random.choice(tasks)
        question = random_task['question']

        # Initialize RAG
        retriever = st.session_state.vectors.as_retriever()
        # language_model = ChatOpenAI(model="gpt-4", temperature=0.0, max_tokens=3000)
        language_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

        # Create a prompt for generating similar questions
        prompt = f"Based on the following question, generate similar questions:\n{question}"

        # Create a chain for RAG
        rag_chain = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever
        )

        # Use RAG to generate new questions
        rag_chain.run(prompt)  # This call generates similar questions

        return question  # Only return the original question
    else:
        return None  # No questions found or tasks are not yet ready.

# Function to transcribe uploaded audio using OpenAI Whisper (working version)
def transcribe_audio(file):
    if file:
        # Save the uploaded file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(file.read())
            temp_audio_path = temp_audio_file.name

        try:
            # Open the temporary audio file and send it to OpenAI's transcription API
            with open(temp_audio_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)

            # Clean up the temporary file
            os.remove(temp_audio_path)

            return transcription['text']  # Return the transcription result
        except Exception as e:
            st.write(f"Error in transcription: {e}")  # Debugging error message
            return None
    return None

# Function to display Speaking Part 2 content
def display_speaking2_content():
    st.title("IELTS Speaking Part 2 Question Generator with RAG")

    # Initialize embeddings and vector database on app start for Speaking Part 2
    if "speaking2_tasks" not in st.session_state:
        create_vector_embedding_for_speaking_part2()

    # Button to generate original question
    if st.button("Generate Random Speaking Part 2 Question"):
        original_question = generate_similar_questions_using_rag()
        if original_question:
            st.session_state.original_question = original_question
        else:
            st.write("No questions found or tasks are not yet ready.")

    # Display original question only
    if 'original_question' in st.session_state:
        st.write(f"*Question:* {st.session_state.original_question}")

        # Audio file uploader for user to attach an audio response
        audio_file = st.file_uploader("Upload your audio response (MP3 format)", type=["mp3"])

        # Transcribe audio when a file is uploaded
        if audio_file is not None:
            transcription = transcribe_audio(audio_file)
            if transcription:
                st.write(f"**Transcription:** {transcription}")
            else:
                st.write("Failed to transcribe the audio. Please try again.")

# Call the main function to display content
display_speaking2_content()
