import json
import re
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
        f"Provide detailed feedback on the following English grammar task:\n{task_text}",
    )
    return response.content

def generate_score(task_text):
    """Generate a score for the grammar task."""
    response = client.invoke(
        f"Evaluate the quality of the following English grammar task and provide a score from 1 to 10:\n{task_text}",
    )
    return response.content

# Helper function to parse JSON response
def parse_json_response(content):
    cleaned_content = re.search(r'\{.*\}', content, re.DOTALL)
    if cleaned_content:
        json_content = cleaned_content.group(0)
        return json.loads(json_content)
    raise HTTPException(status_code=500, detail="Invalid JSON format returned from task generator")

# Separate functions for each topic
def past_time_task():
    response = client.invoke(
        """
        Create a comprehensive grammar task focused on past time forms in JSON format.
        Include keys for "task" (task name), "description" (instructions), "questions" (a list of 15-20 questions), and "answers" (a list of correct answers as lists of strings).
        Cover aspects such as past simple, past continuous, past perfect simple, past perfect continuous, and expressions with 'would,' 'used to,' and 'be/get used to.'
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Past Time Forms Task"),
        "description": parsed_content.get("description", "Answer the following questions using the correct past time forms."),
        "questions": [q if isinstance(q, str) else str(q) for q in parsed_content.get("questions", [])],
        "answers": [
            [ans] if isinstance(ans, str) else ans 
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def future_time_task():
    response = client.invoke(
        """
        Create a grammar task focusing on future time, present tenses in time clauses, and prepositions of time and place in JSON format.
        Include keys for "task" (task name), "description" (instructions), "questions" (a list of 15-20 questions as strings), and "answers" (a list of correct answers as lists of strings).
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Future Time and Time Clauses Task"),
        "description": parsed_content.get("description", "Answer the following questions using the appropriate future tense, present tense in time clauses, or prepositions of time and place."),
        "questions": [q if isinstance(q, str) else str(q) for q in parsed_content.get("questions", [])],
        "answers": [
            [ans] if isinstance(ans, str) else ans
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def articles_quantifiers_task():
    response = client.invoke(
        """
        Generate a task on articles, countable and uncountable nouns, and quantifiers in JSON format.
        Include keys for "task" (task name), "description" (instructions), "questions" (a list of 15-20 questions as strings), and "answers" (a list of correct answers as lists of strings).
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Articles and Quantifiers Task"),
        "description": parsed_content.get("description", "Answer the following questions focusing on the correct use of articles, quantifiers, and distinguishing between countable and uncountable nouns."),
        "questions": [q if isinstance(q, str) else str(q) for q in parsed_content.get("questions", [])],
        "answers": [
            [ans] if isinstance(ans, str) else ans
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def conditionals_task():
    response = client.invoke(
        """
        Create a detailed exercise on conditionals in JSON format.
        Include a key "task" describing the topic, "description" for instructions, "questions" as a list of 15-20 lists, where each list contains strings for the question components, and "answers" as a list of correct answers as lists of strings.
        Cover different conditional types (zero, first, second, third, mixed, and inverted) as well as expressions like 'unless,' 'in case,' 'as/so long as,' and 'provided that.'
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Conditionals Task"),
        "description": parsed_content.get("description", "Answer the following questions, covering different types of conditional statements."),
        "questions": [
            q if isinstance(q, list) else [str(q)]
            for q in parsed_content.get("questions", [])
        ],
        "answers": [
            a if isinstance(a, list) else [str(a)]
            for a in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def comparatives_superlatives_task():
    response = client.invoke(
        """
        Generate a task on comparatives and superlatives in JSON format.
        Include a key "task" describing the topic, "description" for instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Comparatives and Superlatives Task"),
        "description": parsed_content.get("description", "Choose the correct form of the adjective in each question."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(a) for a in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def modals_task():
    response = client.invoke(
        """
        Create a comprehensive exercise on modals in JSON format.
        Include a key "task" describing the topic, "description" for instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Modals Task"),
        "description": parsed_content.get("description", "Choose the correct modal verb to complete each sentence."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in a] if isinstance(a, list) else [str(a)]
            for a in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def passive_causative_task():
    response = client.invoke(
        """
        Generate a task focusing on passive and causative forms in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Passive and Causative Forms"),
        "description": parsed_content.get("description", "Complete each sentence by using the correct passive or causative form."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in a] if isinstance(a, list) else [str(a)]
            for a in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def compound_future_task():
    response = client.invoke(
        """
        Create an exercise on compound future tenses in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Compound Future Tenses"),
        "description": parsed_content.get("description", "Complete each sentence using the appropriate compound future tense."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in a] if isinstance(a, list) else [str(a)]
            for a in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def quantity_task():
    response = client.invoke(
        """
        Generate a task on quantity expressions in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Quantity Expressions"),
        "description": parsed_content.get("description", "Complete each sentence with the correct quantity expression."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(answer) for answer in parsed_content.get("answers", [])
        ]
    }
    
    return task_data



def passive_structures_task():
    response = client.invoke(
        """
        Create a detailed exercise on passive structures in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Passive Structures"),
        "description": parsed_content.get("description", "Rewrite each sentence in the passive voice as instructed."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(answer) for answer in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def uses_of_it_task():
    response = client.invoke(
        """
        Generate a task with 15-20 questions on the various uses of 'it' in English in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Uses of 'It' in English"),
        "description": parsed_content.get("description", "Identify and correctly use different forms of 'it' in each sentence."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in ans] if isinstance(ans, list) else [str(ans)]
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def relative_clauses_task():
    response = client.invoke(
        """
        Create a detailed task on relative clauses and reduced relative clauses in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Relative Clauses and Reduced Relative Clauses"),
        "description": parsed_content.get("description", "Use the correct relative clause or reduced form to complete each sentence."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(ans) for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def modals_speculation_task():
    response = client.invoke(
        """
        Generate a task on modal verbs of speculation in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Modal Verbs: Speculation"),
        "description": parsed_content.get("description", "Use the appropriate modal verb to express speculation for each sentence."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in ans] if isinstance(ans, list) else [str(ans)]
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def talking_about_ability_task():
    response = client.invoke(
        """
        Create a task on discussing ability in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Talking about Ability"),
        "description": parsed_content.get("description", "Use the appropriate form to express ability for each sentence."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            [str(answer) for answer in ans] if isinstance(ans, list) else [str(ans)]
            for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def emphatic_forms_task():
    response = client.invoke(
        """
        Generate a task on emphatic forms in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Ensure the required structure
    task_data = {
        "task": parsed_content.get("task", "Emphatic Forms"),
        "description": parsed_content.get("description", "Complete each sentence using the correct emphatic form."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(ans) for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


def wh_words_task():
    response = client.invoke(
        """
        Create an exercise on the use of 'whatever,' 'whoever,' 'whenever,' 'whichever,' 'wherever,' and 'however' in JSON format.
        Include a key "task" describing the topic, "description" with task instructions, "questions" as a list of 15-20 questions, and "answers" as a list of correct answers matching each question.
        """
    )
    # Parse the response content
    parsed_content = parse_json_response(response.content)
    
    # Structure the task data
    task_data = {
        "task": parsed_content.get("task", "WH Words Exercise"),
        "description": parsed_content.get("description", "Complete each sentence using the correct WH word form."),
        "questions": [
            str(q) for q in parsed_content.get("questions", [])
        ],
        "answers": [
            str(ans) for ans in parsed_content.get("answers", [])
        ]
    }
    
    return task_data


# Dictionary mapping topics to functions
topic_functions = {
    "Past Time": past_time_task,
    "Future Time and Present Tenses in Time Clauses": future_time_task,
    "Articles and Quantifiers": articles_quantifiers_task,
    "Conditionals": conditionals_task,
    "Comparatives and Superlatives": comparatives_superlatives_task,
    "Modals": modals_task,
    "Passive and Causative Forms": passive_causative_task,
    "Compound Future Tenses": compound_future_task,
    "Quantity": quantity_task,
    "Passive Structures": passive_structures_task,
    "Uses of It": uses_of_it_task,
    "Relative Clauses and Reduced Relative Clauses": relative_clauses_task,
    "Modal Verbs: Speculation": modals_speculation_task,
    "Talking about Ability": talking_about_ability_task,
    "Emphatic Forms": emphatic_forms_task,
    "Whatever, Whoever, Whenever, Whichever, Wherever, and However": wh_words_task
}

# Initialize session state for the topic
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "Past Time"

# Dropdown menu for selecting a topic
topic = st.selectbox("Choose a grammar topic to practice:", list(topic_functions.keys()))

if st.button("Generate Task"):
    st.session_state.selected_topic = topic
    st.session_state.task_question = topic_functions[topic]()
    st.write(st.session_state.task_question)

# Input for grammar task
task_text = st.text_area("Enter your answer here:")

# Create buttons for Feedback/Score on the same row
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Get Feedback and Score"):
        if task_text:
            feedback = generate_feedback(task_text)
            score = generate_score(task_text)
            st.subheader("Feedback")
            st.write(feedback)
            st.subheader("Score")
            st.write(score)
        else:
            st.warning("Please enter your answers to receive feedback and score.")
