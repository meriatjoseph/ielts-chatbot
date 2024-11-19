import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

# Initialize Streamlit session state
def initialize_state():
    if "generated_topics" not in st.session_state:
        st.session_state.generated_topics = []  # Newly generated topics
    if "questions_by_topic" not in st.session_state:
        st.session_state.questions_by_topic = {}  # Questions for each generated topic
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None  # Currently selected topic
    if "current_questions" not in st.session_state:
        st.session_state.current_questions = []  # Questions for the selected topic

# Load and process the PDF
def load_pdf():
    if "vectors" not in st.session_state:
        st.write("Loading and embedding PDF...")
        loader = PyPDFLoader("speaking3_pdf/Part_3_Speaking.pdf")
        docs = loader.load()

        # Split and embed the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
        st.write("PDF loaded and embedded successfully!")

# Generate new topics dynamically using RAG
def generate_new_topics():
    retriever = st.session_state.vectors.as_retriever()
    # Retrieve context from the document
    retrieved_docs = retriever.get_relevant_documents("IELTS Speaking Part 3 topics")
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Use a language model to generate new topics
    language_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200,
    )
    prompt = (
        f"Based on the following IELTS Speaking Part 3 context:\n"
        f"{context}\n\n"
        f"Generate 5 unique and creative topics for IELTS Speaking Part 3 that are relevant to the content. "
        f"Write each topic as a single line."
    )
    response = language_model.predict(prompt)
    new_topics = [topic.strip() for topic in response.strip().split("\n") if topic.strip()]
    st.session_state.generated_topics = new_topics[:5]
    st.write(f"Generated Topics: {st.session_state.generated_topics}")

# Generate questions for a specific topic
def generate_questions_for_topic(topic):
    retriever = st.session_state.vectors.as_retriever()
    context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])

    # Use the language model to generate questions
    language_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300,
    )
    prompt = (
        f"Based on the topic '{topic}' and the following context:\n"
        f"{context}\n\n"
        f"Generate 10 related questions for this topic. Write each question as a bullet point."
    )
    response = language_model.predict(prompt)
    questions = [line.strip() for line in response.strip().split("\n") if line.strip().startswith("•")]
    st.session_state.questions_by_topic[topic] = questions

# Display the questions for the selected topic
def display_questions(topic):
    if topic in st.session_state.questions_by_topic:
        questions = st.session_state.questions_by_topic[topic]
        st.write(f"### Questions for Topic: {topic}")
        for i, question in enumerate(questions, 1):
            st.write(f"{i}. {question}")
    else:
        st.warning("No questions available for this topic. Please generate questions.")

# Streamlit App
def display_app():
    st.title("IELTS Speaking Part 3: Topic and Question Generator")
    initialize_state()
    load_pdf()

    # Generate new topics if not already done
    if not st.session_state.generated_topics:
        generate_new_topics()

    # Dropdown to select a generated topic
    selected_topic = st.selectbox("Select a Generated Topic", st.session_state.generated_topics)

    # Handle topic selection
    if selected_topic:
        st.session_state.current_topic = selected_topic

        # Generate questions for the selected topic if not already done
        if selected_topic not in st.session_state.questions_by_topic:
            generate_questions_for_topic(selected_topic)

        # Display questions for the selected topic
        display_questions(selected_topic)

# Run the app
if __name__ == "__main__":
    display_app()
