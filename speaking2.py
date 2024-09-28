import os
import random
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
from fastapi import FastAPI, HTTPException

api_app = FastAPI()

def create_vector_embedding_for_speaking_part2():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFLoader("speaking2_pdf/100 IELTS Speaking Topics.pdf")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.session_state.speaking2_tasks = extract_speaking2_questions(st.session_state.final_documents)

def extract_speaking2_questions(documents):
    tasks = []
    for doc in documents:
        content = doc.page_content
        if not content:
            continue
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)
                if len(parts) == 2:
                    question = parts[1].strip()
                    tasks.append({"question": question})
    return tasks

def generate_similar_questions_using_rag():
    if "speaking2_tasks" in st.session_state and st.session_state.speaking2_tasks:
        random_task = random.choice(st.session_state.speaking2_tasks)
        question = random_task['question']
        retriever = st.session_state.vectors.as_retriever()
        language_model = ChatOpenAI()
        prompt = f"Generate similar questions based on the following question:\n{question}"
        rag_chain = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever
        )
        rag_chain.run(prompt)
        return question
    else:
        return None
    
# import uvicorn
# uvicorn.run(api_app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    # Run as a Streamlit app
    # display_writing1_content()

    # Run as a FastAPI app
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)