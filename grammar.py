import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
from google.api_core.exceptions import ResourceExhausted  # Import the exception

# Load environment variables from a .env file
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Google GenAI client
client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def retry_api_call(api_function, retries=3, delay=60):
    """Utility function to retry API calls in case of ResourceExhausted error."""
    for attempt in range(retries):
        try:
            return api_function()
        except ResourceExhausted as e:
            st.warning(f"Quota exceeded, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    st.error("API quota exceeded. Please try again later.")
    return "Quota exceeded."

def generate_grammar_task():
    """Generate a grammar practice task for IELTS General along with correct answers."""
    return retry_api_call(
        lambda: client.invoke(
            "Generate a grammar task focused on IELTS General, including common grammar topics such as tenses, articles, and conditionals. Also, provide the correct answers for the task."
        ).content
    )

def generate_feedback(task_text):
    """Provide feedback for a grammar task response."""
    return retry_api_call(
        lambda: client.invoke(
            f"Provide detailed feedback on the following IELTS General grammar practice task:\n{task_text}"
        ).content
    )

def generate_score(task_text):
    """Generate a score based on the grammar task response."""
    return retry_api_call(
        lambda: client.invoke(
            f"Evaluate the quality of the response for the following IELTS General task and provide a score from 1 to 10:\n{task_text}"
        ).content
    )

def display_grammar():
    """Display the grammar task interface."""
    st.title("IELTS Grammar Practice Task")

    # Initialize session state for grammar task
    if 'grammar_task' not in st.session_state:
        st.session_state.grammar_task = generate_grammar_task()

    # Process the grammar task and extract the answers if present
    if "Correct Answers:" in st.session_state.grammar_task:
        grammar_task, grammar_answers = st.session_state.grammar_task.split("Correct Answers:")
    else:
        grammar_task = st.session_state.grammar_task
        grammar_answers = "Correct answers not provided."

    st.subheader("Grammar Task")
    st.write(grammar_task)

    # Display correct answers in green color
    st.subheader("Correct Answers")
    st.markdown(f"<span style='color:green;'>{grammar_answers.strip()}</span>", unsafe_allow_html=True)

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
