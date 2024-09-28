import os
import random
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, HTTPException
api_app = FastAPI()

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
        if not content:  # If there's no content, skip to the next document
            continue

        # Split the content into lines
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            # Check if the line starts with a number (e.g., "1. Describe an internet business...")
            if line and line[0].isdigit() and "." in line:
                # Extract the question
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
        language_model = ChatOpenAI()

        # Create a prompt for generating similar questions (not stored)
        prompt = f"Based on the following question, generate similar questions:\n{question}"

        # Create a chain for RAG
        rag_chain = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever
        )

        # Use RAG to generate new questions (not displayed)
        rag_chain.run(prompt)  # This call generates similar questions without displaying them

        return question  # Only return the original question
    else:
        return None  # No questions found or tasks are not yet ready.

# Initialize embeddings and vector database on app start for Speaking Part 2
if "speaking2_tasks" not in st.session_state:
    create_vector_embedding_for_speaking_part2()

# Streamlit UI for Speaking Part 2
st.title("IELTS Speaking Part 2 Question Generator with RAG")

# Button to generate original question
if st.button("Generate Random Speaking Part 2 Question"):
    original_question = generate_similar_questions_using_rag()
    if original_question:
        st.session_state.original_question = original_question
    else:
        st.write("No questions found or tasks are not yet ready.")

# Display original question only
if 'original_question' in st.session_state:
    st.write(f"**Question:** {st.session_state.original_question}")

    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)