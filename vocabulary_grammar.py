import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables from a .env file
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Initialize Google GenAI client
client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def generate_vocabulary_task():
    """Generate a vocabulary practice task for IELTS General."""
    response = client.invoke(
        "Generate a vocabulary task focused on IELTS General, including words frequently used in IELTS and relevant writing exercises."
    )
    return response.content

def generate_grammar_task():
    """Generate a grammar practice task for IELTS General."""
    response = client.invoke(
        "Generate a grammar task focused on IELTS General, including common grammar topics such as tenses, articles, and conditionals."
    )
    return response.content

def generate_feedback(task_text):
    """Provide feedback for a grammar or vocabulary task response."""
    response = client.invoke(
        f"Provide detailed feedback on the following IELTS General grammar or vocabulary practice task:\n{task_text}",
    )
    return response.content

def generate_score(task_text):
    """Generate a score based on the vocabulary or grammar task response."""
    response = client.invoke(
        f"Evaluate the quality of the response for the following IELTS General task and provide a score from 1 to 10:\n{task_text}",
    )
    return response.content

def display_vocabulary_grammar():
    """Display the vocabulary and grammar practice sections in Streamlit."""
    # Initialize session state for vocabulary and grammar tasks
    if 'vocabulary_task' not in st.session_state:
        st.session_state.vocabulary_task = generate_vocabulary_task()

    if 'grammar_task' not in st.session_state:
        st.session_state.grammar_task = generate_grammar_task()

    # Tab-based interface for Vocabulary and Grammar
    tab1, tab2 = st.tabs(["Vocabulary Practice", "Grammar Practice"])

    # Vocabulary Practice Tab
    with tab1:
        st.subheader("Vocabulary Task")
        st.write(st.session_state.vocabulary_task)

        # Input for vocabulary task response
        vocab_task_text = st.text_area("Enter your response to the vocabulary task here:")

        # Buttons for feedback and score
        if st.button("Get Vocabulary Feedback and Score"):
            if vocab_task_text:
                vocab_feedback = generate_feedback(vocab_task_text)
                vocab_score = generate_score(vocab_task_text)
                st.subheader("Vocabulary Feedback")
                st.write(vocab_feedback)
                st.subheader("Vocabulary Score")
                st.write(vocab_score)
            else:
                st.warning("Please enter a response to receive feedback and score.")

        # Button for next vocabulary task
        if st.button("Next Vocabulary Task"):
            st.session_state.vocabulary_task = generate_vocabulary_task()
            st.rerun()  # Refresh the page for the new vocabulary task

    # Grammar Practice Tab
    with tab2:
        st.subheader("Grammar Task")
        st.write(st.session_state.grammar_task)

        # Input for grammar task response
        grammar_task_text = st.text_area("Enter your response to the grammar task here:")

        # Buttons for feedback and score
        if st.button("Get Grammar Feedback and Score"):
            if grammar_task_text:
                grammar_feedback = generate_feedback(grammar_task_text)
                grammar_score = generate_score(grammar_task_text)
                st.subheader("Grammar Feedback")
                st.write(grammar_feedback)
                st.subheader("Grammar Score")
                st.write(grammar_score)
            else:
                st.warning("Please enter a response to receive feedback and score.")

        # Button for next grammar task
        if st.button("Next Grammar Task"):
            st.session_state.grammar_task = generate_grammar_task()
            st.rerun()  # Refresh the page for the new grammar task
