import os
import random
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI  # Correct import for ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import HumanMessage  # Import HumanMessage to create a message
import requests
import time

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPEN_API_KEY')

# Initialize LLM with Chat Model
llm = ChatOpenAI(model="gpt-4", api_key=open_api_key)

# Define the prompt template for generating IELTS writing tasks
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Based on the content provided, generate a random IELTS Writing Task 2 prompt.
    Do not include any answers or explanations, just the task prompt.
    
    <context>
    {context}
    </context>

    Task: Generate an IELTS Writing Task 2 prompt."""
)

# Function to create vector embeddings and load documents
def create_vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("task2_writing")  # Data ingestion
    st.session_state.docs = st.session_state.loader.load()  # Document loading
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.session_state.writing_tasks = extract_writing_tasks(st.session_state.final_documents)

# Function to extract IELTS writing tasks from documents
def extract_writing_tasks(documents):
    tasks = []
    for doc in documents:
        if "Task 2" in doc.page_content:  # Assuming that "Task 2" indicates a writing prompt
            tasks.append(doc.page_content)
    return tasks

# Function to generate a random question using the LLM
def generate_llm_question():
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        document_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        # Properly format the messages for the chat endpoint
        response = llm([HumanMessage(content="Generate a random IELTS Writing Task")])
        
        # Access the content directly from the response
        if response:
            # Directly use the content of the AIMessage object
            answer = response.content
        else:
            answer = 'No answer found in the response.'

        return answer
    else:
        return "Vector embeddings are not yet ready."


# Function to check grammar using LanguageTool
def check_grammar_with_languagetool(text):
    url = "https://api.languagetool.org/v2/check"
    payload = {
        "text": text,
        "language": "en-US"
    }
    response = requests.post(url, data=payload)
    response_json = response.json()
    
    corrected_text, band_score, scores = apply_corrections(text, response_json)
    return corrected_text, band_score, scores, response_json

# Function to apply corrections and calculate band score
def apply_corrections(text, response_json):
    matches = response_json.get('matches', [])
    corrected_text = text
    total_errors = len(matches)
    task_response_score = 9  # Start with full points for each criterion
    lexical_resource_score = 9
    grammatical_range_and_accuracy_score = 9

    # Apply corrections in reverse order to prevent indexing issues
    for match in sorted(matches, key=lambda x: x['offset'], reverse=True):
        offset = match['offset']
        length = match['length']
        replacements = match.get('replacements', [])
        
        if replacements:
            best_replacement = replacements[0]['value']  # Take the most confident suggestion
            corrected_text = corrected_text[:offset] + best_replacement + corrected_text[offset + length:]
        
        # Deduct points based on the type of error
        issue_type = match.get('rule', {}).get('issueType')
        if issue_type == 'misspelling':
            lexical_resource_score -= 0.5  # Deduct 0.5 points for each misspelling
        elif issue_type == 'grammar':
            grammatical_range_and_accuracy_score -= 0.5  # Deduct 0.5 points for each grammar issue
        elif issue_type == 'punctuation':
            grammatical_range_and_accuracy_score -= 0.25  # Deduct 0.25 points for each punctuation issue
        else:
            lexical_resource_score -= 0.25  # Deduct 0.25 points for other issues

    # Ensure the scores do not go below 0 (minimum score)
    task_response_score = max(task_response_score, 0)
    lexical_resource_score = max(lexical_resource_score, 0)
    grammatical_range_and_accuracy_score = max(grammatical_range_and_accuracy_score, 0)

    # Calculate the band score as an average of all three components
    band_score = (task_response_score + lexical_resource_score + grammatical_range_and_accuracy_score) / 3

    # Return the corrected text, band score, and individual component scores
    return corrected_text, band_score, {
        'Task Response': task_response_score,
        'Lexical Resource': lexical_resource_score,
        'Grammatical Range and Accuracy': grammatical_range_and_accuracy_score
    }

# Display function to be called from app.py
def display_writing2_content():
    st.title("IELTS Writing Task Generator")
    
    # Initialize embeddings and vector database on app start
    if "vectors" not in st.session_state:
        create_vector_embedding()

    # Streamlit UI
    #st.title("IELTS Writing Task Generator and Analyzer")

    # Display a single random Task 2 question and update only on button click
    if 'question_generated' not in st.session_state:
        st.session_state.question_generated = False

    def update_question():
        st.session_state.current_task = generate_llm_question()
        st.session_state.question_generated = True

    if st.button("Generate Random Writing Task"):
        update_question()

    if st.session_state.question_generated:
        st.write("Random IELTS Writing Task 2 Question:")
        st.write(st.session_state.current_task)

    # User inputs their answer
    user_answer = st.text_area("Enter your answer:")

    if user_answer:
        corrected_text, band_score, scores, grammar_result = check_grammar_with_languagetool(user_answer)
        
        # Print the grammar result for debugging
        #st.write("Grammar check result:")
        #st.json(grammar_result)
        
        st.write("Corrected Text:")
        st.write(corrected_text)
        
        # Check if the score is below 6
        if band_score < 6:
            st.write("Band Score: Test Failed")
        else:
            st.write(f"Band Score: {band_score:.1f}")

        st.write("Individual Scores:")
        st.write(scores)
        
        if 'matches' in grammar_result:
            matches = grammar_result['matches']
            if matches:
                st.write("Here are some suggestions to improve your answer:")
                for match in matches:
                    st.write(f"Error: {match['context']['text']}")
                    st.write(f"Suggestion: {', '.join([r['value'] for r in match['replacements']])}")
                    st.write(f"Message: {match['message']}")
                    st.write("---------------")
            else:
                st.write("No grammar issues found.")
        else:
            st.write("Unexpected grammar check result format.")
