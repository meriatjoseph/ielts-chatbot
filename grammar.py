import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
from google.api_core.exceptions import ResourceExhausted  # Import the exception
import json

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

import re

def generate_grammar_task():
    """Generate a grammar practice task for IELTS General along with correct answers in JSON format."""
    result = retry_api_call(
        lambda: client.invoke(
            "Generate a grammar task focused on IELTS General, including common grammar topics such as tenses, articles, and conditionals. "
            "Provide the gaps (questions) and answers with explanations in a structured JSON format."
        ).content
    )

    # Print the raw result for debugging
    st.write("Raw result from API:", result)

    # Clean up the result to ensure valid JSON
    cleaned_result = re.sub(r'```json|```', '', result).strip()  # Remove ```json and any other unwanted characters

    # Print cleaned result
    st.write("Cleaned result:", cleaned_result)

    # Try to parse the cleaned result as JSON to ensure it's in dictionary format
    try:
        parsed_result = json.loads(cleaned_result)  # Ensure the result is parsed as a JSON dictionary
        if not isinstance(parsed_result, dict):
            st.error("The parsed result is not a dictionary.")
            raise ValueError("Parsed result is not a dictionary.")
        return parsed_result
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        raise ValueError("Unable to parse the response as JSON.") from e



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

    # Display raw response to debug JSON structure issues
    st.subheader("Raw API Response (for debugging)")
    st.write(st.session_state.grammar_task)  # This should now display the parsed dictionary

    # Try parsing the response as JSON (it should already be a dictionary now)
    try:
        grammar_data = st.session_state.grammar_task
        
        # Check if it's really a dictionary
        if not isinstance(grammar_data, dict):
            st.error("The grammar task is not in dictionary format. Please check.")
            return
        
        # Extract questions, answers, and gaps
        gaps = {str(q['id']): q['options'] for q in grammar_data['questions']}
        answers = {str(q['id']): [q['answer'], q['explanation']] for q in grammar_data['questions']}
        text = " ".join([q['text'] for q in grammar_data['questions']])  # Combine all questions into a single string

        # Display the grammar task text
        st.subheader("Grammar Task Text")
        st.write(text)

        # Display the grammar task with gaps
        st.subheader("Grammar Questions and Options")
        for gap_num, options in gaps.items():
            st.write(f"Question {gap_num}:")
            for i, option in enumerate(options):
                st.write(f"{chr(65+i)}. {option}")

        # Display correct answers in green color
        st.subheader("Correct Answers")
        for gap_num, answer_data in answers.items():
            st.markdown(f"**Question {gap_num}:** Correct answer is **{answer_data[0]}**. Explanation: {answer_data[1]}")

        # Display the generated JSON data
        st.subheader("Generated JSON Data")
        st.json(grammar_data)

    except KeyError:
        st.error("Error accessing the grammar task data. Please try again.")
        return

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

# Call display_grammar to render the UI
if __name__ == "__main__":
    display_grammar()