import os
import random
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import streamlit as st
from fastapi import FastAPI, HTTPException

api_app = FastAPI()

def extract_topics_and_questions():
    if "topics_with_questions" not in st.session_state:
        loader = PyPDFLoader("speaking3_pdf/Part_3_Speaking.pdf")
        st.session_state.docs = loader.load()
        st.session_state.final_documents = st.session_state.docs
        full_text = "".join([doc.page_content for doc in st.session_state.docs])
        st.session_state.topics_with_questions = process_topics_and_questions(full_text)

def process_topics_and_questions(pdf_content):
    topics_with_questions = {}
    current_topic = None
    lines = pdf_content.split("\n")

    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            current_topic = line
            topics_with_questions[current_topic] = []
        elif current_topic and line.startswith("•"):
            topics_with_questions[current_topic].append(line)

    return topics_with_questions

def generate_questions_using_rag(selected_topic):
    language_model = ChatOpenAI(api_key=os.getenv('OPEN_API_KEY'))
    if "topics_with_questions" in st.session_state:
        questions = st.session_state.topics_with_questions[selected_topic]
        retriever = st.session_state.vectors.as_retriever()
        retrieved_content = retriever.get_relevant_documents(selected_topic)
        prompt = f"Generate new questions based on the topic '{selected_topic}' and context:\n" + "\n".join(questions)
        rag_chain = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever
        )
        generated_questions_response = rag_chain.run(prompt)
        generated_questions = str(generated_questions_response).strip().split('\n')
        all_questions = questions + generated_questions
        random.shuffle(all_questions)
        return random.choice(all_questions)
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