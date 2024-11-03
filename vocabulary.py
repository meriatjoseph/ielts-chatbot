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
st.markdown('<p class="title">English Vocabulary Task Assistant</p>', unsafe_allow_html=True)

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
        Include the following keys:
        - "task": a string that provides the task name.
        - "description": a string describing the instructions (e.g., 'Complete the sentences with one word in each gap.').
        - "questions": a list of sentences as strings, each requiring a one-word answer.
        - "answers": a list of strings with the correct word matching each question.
        Ensure the JSON response adheres strictly to these key names and formats.
        """
    )
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure required keys exist with correct fallback values
    task = task_data.get("task", "Complete the Sentence")
    description = task_data.get("description", "Complete the sentences with one word in each gap.")
    
    # Ensure 'questions' and 'answers' are lists of strings
    questions = [
        str(q) for q in task_data.get("questions", []) if isinstance(q, str)
    ]
    answers = [
        str(a) for a in task_data.get("answers", []) if isinstance(a, str)
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }

def error_correction_task():
    response = client.invoke(
        """
        Generate a vocabulary task called "Find and Correct the Mistake" in JSON format with the following structure:
        - "task": a string providing the task name, such as "Find and Correct the Mistake."
        - "description": a string with the instructions (e.g., 'Identify and correct the vocabulary mistake in each sentence.').
        - "questions": a list of sentences containing vocabulary errors, where each question requires a correction.
        - "answers": a list of objects, where each object contains:
            - "incorrect": the incorrect word
            - "correct": the correct replacement.
        Ensure the JSON response strictly adheres to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with default values if missing
    task = task_data.get("task", "Find and Correct the Mistake")
    description = task_data.get("description", "Identify and correct the vocabulary mistake in each sentence.")
    
    # Process 'questions' into a numbered dictionary
    questions = {
        str(i + 1): str(q) for i, q in enumerate(task_data.get("questions", [])) if isinstance(q, str)
    }

    # Process 'answers' to ensure each entry contains 'incorrect' and 'correct'
    answers = [
        {
            "incorrect": answer.get("incorrect", ""),
            "correct": answer.get("correct", "")
        }
        for answer in task_data.get("answers", [])
        if isinstance(answer, dict) and "incorrect" in answer and "correct" in answer
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }



def multiple_choice_task():
    response = client.invoke(
        """
        Create a vocabulary task called "Choose the Correct Option" in JSON format with the following structure:
        - "task": a string providing the task name, such as "Choose the Correct Option."
        - "description": a string with the instructions (e.g., 'Select the most appropriate word to complete each sentence.').
        - "questions": a list of dictionaries, each containing:
            - "question": the sentence text with a blank
            - "options": a list of two to three vocabulary options to choose from.
        - "answers": a list of objects, each containing:
            - "question_no": the number of the question
            - "answer": the correct option for that question.
        Ensure the JSON response strictly adheres to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Set task and description with defaults if missing
    task = task_data.get("task", "Choose the Correct Option")
    description = task_data.get("description", "Select the most appropriate word to complete each sentence.")
    
    # Process 'questions' into a numbered dictionary
    questions = {
        str(i + 1): {
            "question": q.get("question", ""),
            "options": q.get("options", [])
        }
        for i, q in enumerate(task_data.get("questions", []))
        if isinstance(q, dict) and "question" in q and "options" in q
    }

    # Process 'answers' to ensure each entry contains 'question_no' and 'answer'
    answers = [
        {
            "question_no": str(i + 1),
            "answer": ans.get("answer", "")
        }
        for i, ans in enumerate(task_data.get("answers", []))
        if isinstance(ans, dict) and "answer" in ans
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }

def synonyms_antonyms_task():
    response = client.invoke(
        """
        Generate a vocabulary task focused on synonyms and antonyms in JSON format with the following structure:
        - "task": a string that specifies the task name, such as "Synonyms and Antonyms."
        - "description": a string with the instructions (e.g., 'Select the correct synonym or antonym for the underlined word.').
        - "questions": a list of dictionaries, each containing:
            - "sentence": the sentence with an underlined word to be replaced.
            - "options": a list of vocabulary options (synonyms or antonyms).
            - "answer": the correct option among the provided choices.
            - "answertype": indicates whether the correct option is a "synonym" or "antonym".
        Ensure the JSON response adheres strictly to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Synonyms and Antonyms")
    description = task_data.get("description", "Select the correct synonym or antonym for the underlined word.")
    
    # Process 'questions' to ensure each entry includes 'sentence', 'options', 'answer', and 'answertype'
    questions = [
        {
            "sentence": q.get("sentence", ""),
            "options": q.get("options", []),
            "answer": q.get("answer", ""),
            "answertype": q.get("answertype", "synonym or antonym")
        }
        for q in task_data.get("questions", [])
        if isinstance(q, dict) and "sentence" in q and "options" in q and "answer" in q and "answertype" in q
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions
    }


def collocations_task():
    response = client.invoke(
        """
        Create a vocabulary task on collocations in JSON format with the following structure:
        - "task": a string providing the task name, such as "Collocations Task."
        - "description": a string with instructions (e.g., 'Complete each sentence with the appropriate collocation.').
        - "questions": a list of sentences as strings, each with a blank for the correct collocation.
        - "answers": a list of lists, where each inner list contains possible correct collocations for each corresponding question.
        Ensure the JSON response strictly adheres to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Collocations Task")
    description = task_data.get("description", "Complete each sentence with the appropriate collocation.")
    
    # Process 'questions' as a list of strings
    questions = [
        str(q) for q in task_data.get("questions", []) if isinstance(q, str)
    ]
    
    # Process 'answers' as a list of lists of strings
    answers = [
        [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
        for answer in task_data.get("answers", [])
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }



def word_forms_task():
    response = client.invoke(
        """
        Generate a vocabulary task called "Word Forms" in JSON format with the following structure:
        - "task": a string that provides the task name, such as "Word Forms Task."
        - "description": a string with the instructions (e.g., 'Provide the correct form of the word to complete each sentence.').
        - "questions": a list of sentences as strings, each needing a word form.
        - "answers": a list of strings where each entry is the correct form matching the corresponding question.
        Ensure the JSON response strictly adheres to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Word Forms Task")
    description = task_data.get("description", "Provide the correct form of the word to complete each sentence.")
    
    # Process 'questions' as a list of strings
    questions = [
        str(q) for q in task_data.get("questions", []) if isinstance(q, str)
    ]
    
    # Process 'answers' as a list of strings
    answers = [
        str(a) for a in task_data.get("answers", []) if isinstance(a, str)
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }


def context_clues_task():
    response = client.invoke(
        """
        Create a vocabulary task on understanding words from context clues in JSON format with the following structure:
        - "task": a string that provides the task name, such as "Context Clues Task."
        - "description": a string with the instructions (e.g., 'Infer the meaning of the underlined word using context clues.').
        - "questions": a list of sentences as strings, each containing an underlined word.
        - "answers": a list of lists, where each inner list contains possible meanings (as strings) for the underlined word in the corresponding question.
        Ensure the JSON response strictly adheres to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Context Clues Task")
    description = task_data.get("description", "Infer the meaning of the underlined word using context clues.")
    
    # Process 'questions' as a list of strings
    questions = [
        str(q) for q in task_data.get("questions", []) if isinstance(q, str)
    ]
    
    # Process 'answers' as a list of lists of strings
    answers = [
        [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
        for answer in task_data.get("answers", [])
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }


def idioms_phrases_task():
    response = client.invoke(
        """
        Generate a vocabulary task focused on idioms and phrases in JSON format with the following structure:
        - "task": a string providing the task name, such as "Idioms and Phrases Task."
        - "description": a string with instructions (e.g., 'Choose the correct idiom or phrase to complete each sentence.').
        - "questions": a list of dictionaries, each containing:
            - "sentence": a sentence with a blank for an idiom or phrase.
            - "options": a list of idioms or phrases as choices.
        - "answers": a list of strings, each one being the correct idiom or phrase matching the corresponding question.
        Ensure the JSON response adheres strictly to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Idioms and Phrases Task")
    description = task_data.get("description", "Choose the correct idiom or phrase to complete each sentence.")
    
    # Process 'questions' as a list of dictionaries containing 'sentence' and 'options'
    questions = [
        {
            "sentence": q.get("sentence", ""),
            "options": q.get("options", [])
        }
        for q in task_data.get("questions", [])
        if isinstance(q, dict) and "sentence" in q and "options" in q
    ]
    
    # Process 'answers' as a list of strings
    answers = [
        str(a) for a in task_data.get("answers", []) if isinstance(a, str)
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }


def phrasal_verbs_task():
    response = client.invoke(
        """
        Generate a vocabulary task on phrasal verbs in JSON format with the following structure:
        - "task": a string providing the task name, such as "Phrasal Verbs Task."
        - "description": a string with instructions (e.g., 'Choose the correct phrasal verb to complete each sentence.').
        - "questions": a list of dictionaries, each containing:
            - "sentence": a sentence with a blank for a phrasal verb.
            - "options": a list of phrasal verbs as choices.
        - "answers": a list of strings, each one being the correct phrasal verb matching the corresponding question.
        Ensure the JSON response adheres strictly to these key names and formats.
        """
    )
    
    # Parse JSON response content
    task_data = parse_json_response(response.content)
    
    # Ensure 'task' and 'description' are present with defaults if missing
    task = task_data.get("task", "Phrasal Verbs Task")
    description = task_data.get("description", "Choose the correct phrasal verb to complete each sentence.")
    
    # Process 'questions' as a list of dictionaries containing 'sentence' and 'options'
    questions = [
        {
            "sentence": q.get("sentence", ""),
            "options": q.get("options", [])
        }
        for q in task_data.get("questions", [])
        if isinstance(q, dict) and "sentence" in q and "options" in q
    ]
    
    # Process 'answers' as a list of strings
    answers = [
        str(a) for a in task_data.get("answers", []) if isinstance(a, str)
    ]
    
    # Return structured response with consistent keys
    return {
        "task": task,
        "description": description,
        "questions": questions,
        "answers": answers
    }


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
