import json
from fastapi import HTTPException
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

st.markdown("""
    <style>
    .title {
        color: blue;
        font-size: 2.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Streamlit app
st.markdown('<p class="title">English Grammar and Vocabulary Task Assistant</p>', unsafe_allow_html=True)

# Initialize Google GenAI client
client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def generate_feedback(task_text):
    """Generate feedback """
    response = client.invoke(
        f"Provide detailed feedback on the following English vocabulary task:\n{task_text}",
    )
    return response.content

import re

# Separate functions for each vocabulary task type
def sentence_completion_task():
    response = client.invoke(
        """
        Create a vocabulary task called "Complete the Sentence" in JSON format with verified answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences requiring one-word answers, and "answers" with correct words matching each question.
        Use formats such as 'Complete the sentences with one word in each gap.'
        """
    )
    return parse_json_response(response.content)

def error_correction_task():
    response = client.invoke(
        """
        Generate a vocabulary task called "Find and Correct the Mistake" in JSON format with both incorrect and corrected words for each question.
        Include a key "task" describing the topic, "questions" with 15-20 sentences containing vocabulary errors, and "answers" with both the incorrect word (marked) and the correct replacement.
        Ensure each answer clearly shows both the original mistake and the correction.
        """
    )
    return parse_json_response(response.content)

def multiple_choice_task():
    response = client.invoke(
        """
        Create a vocabulary task called "Choose the Correct Option" in JSON format with reliable answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences offering two to three vocabulary choices, and "answers" with the correct option for each question.
        Ensure each correct option is well-suited to the context of the sentence.
        """
    )
    return parse_json_response(response.content)

def synonyms_antonyms_task():
    response = client.invoke(
        """
        Generate a vocabulary task focused on synonyms and antonyms in JSON format, with verified answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences where the word to be replaced is underlined.
        Provide the correct synonym or antonym as the answer.
        Confirm that each answer is accurate and relevant to the underlined word.
        """
    )
    return parse_json_response(response.content)

def collocations_task():
    response = client.invoke(
        """
        Create a vocabulary task on collocations in JSON format with accurate answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences requiring students to complete each sentence with a correct collocation, and "answers" with verified collocation phrases matching each question.
        Double-check that each collocation is correct for the context.
        """
    )
    return parse_json_response(response.content)

def word_forms_task():
    response = client.invoke(
        """
        Generate a vocabulary task called "Word Forms" in JSON format with verified answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences requiring students to provide the correct form of a word, and "answers" with the correct forms matching each question.
        Confirm that each form fits the sentence context and grammar requirements.
        """
    )
    return parse_json_response(response.content)

def context_clues_task():
    response = client.invoke(
        """
        Create a vocabulary task on understanding words from context clues in JSON format.
        In each question, underline the target word in the sentence that needs to be inferred from context.
        Include a key "task" describing the topic, "questions" with 15-20 sentences containing an underlined word, and "answers" with the inferred meaning of each underlined word.
        """
    )
    return parse_json_response(response.content)

def idioms_phrases_task():
    response = client.invoke(
        """
        Generate a vocabulary task focused on idioms and phrases in JSON format with accurate answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences where students need to fill in or choose the correct idiom or phrase, and "answers" with the verified idioms or phrases matching each question.
        Ensure each idiom or phrase fits logically within the sentence context.
        """
    )
    return parse_json_response(response.content)

def phrasal_verbs_task():
    response = client.invoke(
        """
        Create a vocabulary task on phrasal verbs in JSON format with accurate answers.
        Include a key "task" describing the topic, "questions" with 15-20 sentences requiring students to complete each sentence with the correct phrasal verb, and "answers" with verified phrasal verbs matching each question.
        Confirm each phrasal verb fits the context of the sentence.
        """
    )
    return parse_json_response(response.content)

# Helper function to parse JSON response
def parse_json_response(content):
    cleaned_content = re.search(r'\{.*\}', content, re.DOTALL)
    if cleaned_content:
        json_content = cleaned_content.group(0)
        return json.loads(json_content)
    raise HTTPException(status_code=500, detail="Invalid JSON format returned from task generator")

# Function to generate vocabulary task based on task type
def generate_vocabulary_task(task_type):
    task_functions = {
        "Sentence Completion": sentence_completion_task,
        "Error Correction": error_correction_task,
        "Multiple Choice": multiple_choice_task,
        "Synonyms and Antonyms": synonyms_antonyms_task,
        "Collocations": collocations_task,
        "Word Forms": word_forms_task,
        "Context Clues": context_clues_task,
        "Idioms and Phrases": idioms_phrases_task,
        "Phrasal Verbs": phrasal_verbs_task
    }
    return task_functions[task_type]()

# Initialize session state for the vocabulary task
if 'selected_task_type' not in st.session_state:
    st.session_state.selected_task_type = "Sentence Completion"

# Dropdown menu for selecting a vocabulary task type
task_type = st.selectbox("Choose a vocabulary task type to practice:", 
                     ["Sentence Completion", "Error Correction", "Multiple Choice", 
                      "Synonyms and Antonyms", "Collocations", "Word Forms", "Context Clues", 
                      "Idioms and Phrases", "Phrasal Verbs"])

if st.button("Generate Vocabulary Task"):
    st.session_state.selected_task_type = task_type
    st.session_state.vocab_task = generate_vocabulary_task(task_type)
    st.write(st.session_state.vocab_task)

# Input for vocabulary task answers
task_text = st.text_area("Enter your answer here:")

# Create buttons for Feedback/Score on the same row
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Get Feedback and Score"):
        if task_text:
            feedback = generate_feedback(task_text)
            # score = generate_score(task_text)
            st.subheader("Feedback")
            st.write(feedback)
            st.subheader("Score")
            # st.write(score)
        else:
            st.warning("Please enter your answers to receive feedback and score.") 
