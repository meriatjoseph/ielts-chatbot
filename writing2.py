import os
import random
import requests
import bs4
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# FastAPI instance
api_app = FastAPI()

# Pydantic model for request validation
class WritingTaskRequest(BaseModel):
    user_answer: str

# Function to extract IELTS Writing Task 2 questions and sample answers from web documents
def extract_writing_tasks_from_web(urls):
    tasks = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            
            elements = soup.find_all("div", class_="et_pb_section et_pb_section_1 et_section_regular")
            for element in elements:
                task = element.get_text(strip=True)
                if task:
                    sample_answer_parent_div = soup.find("div", class_="et_pb_module et_pb_toggle et_pb_toggle_1 et_pb_toggle_item et_pb_toggle_close")
                    sample_answer_div = sample_answer_parent_div.find("div", class_="et_pb_toggle_content clearfix") if sample_answer_parent_div else None
                    sample_answer = sample_answer_div.get_text(separator=' ', strip=True) if sample_answer_div else None
                    tasks.append({
                        "text": task,
                        "sample_answer": sample_answer,
                        "url": url
                    })
        except requests.exceptions.RequestException as e:
            print(f"Error processing {url}: {e}")
    return tasks

# Function to generate IELTS Writing Test URLs with leading zeros
def generate_ielts_test_urls():
    base_url = "https://ieltstrainingonline.com/ielts-writing-practice-test-"
    urls = [f"{base_url}{i:02d}/" for i in range(1, 10)]  
    return urls

# Generate the test URLs
ielts_test_urls = generate_ielts_test_urls()

# Extract writing tasks from web documents for Writing Task 2
writing_tasks = extract_writing_tasks_from_web(ielts_test_urls)

# Function to generate a random task from extracted tasks
def generate_random_task():
    if writing_tasks:
        return random.choice(writing_tasks)
    else:
        raise HTTPException(status_code=404, detail="No writing tasks available.")

# Function to check the correctness of the user's answer using LLM
def check_answer_correctness(question, user_answer, sample_answer):
    if not user_answer or not sample_answer or not question:
        return "Insufficient data to check correctness."
    messages = [
        SystemMessage(content="You are an expert in evaluating IELTS writing tasks."),
        HumanMessage(content=f"Evaluate the following user's answer for an IELTS writing task based on the provided question and the ideal qualities of a high-quality response. "
                             f"Assess the user's answer for coherence, structure, grammar, vocabulary usage, relevance to the task, and overall quality. "
                             f"Provide feedback highlighting the strengths and areas for improvement in the user's answer.\n\n"
                             f"Question:\n{question}\n\n"
                             f"Ideal qualities for the response include proper grammar, clear structure, and relevance to the question.\n\n"
                             f"User's Answer:\n{user_answer}\n\n"
                             f"Feedback:")
    ]
    response = llm(messages=messages)
    feedback = response.content if response else "Could not generate feedback."
    return feedback

# Function to check grammar using LanguageTool
def check_grammar_with_languagetool(text):
    url = "https://api.languagetool.org/v2/check"
    payload = {
        "text": text,
        "language": "en-US"
    }
    response = requests.post(url, data=payload)
    response_json = response.json()
    corrected_text, band_score = apply_corrections(text, response_json)
    return corrected_text, band_score, response_json

# Function to apply corrections and calculate band score
def apply_corrections(text, response_json):
    matches = response_json.get('matches', [])
    corrected_text = text
    total_errors = len(matches)
    task_response_score = 9  # Start with full points for each criterion
    coherence_and_cohesion_score = 9
    lexical_resource_score = 9
    grammatical_range_and_accuracy_score = 9

    for match in sorted(matches, key=lambda x: x['offset'], reverse=True):
        offset = match['offset']
        length = match['length']
        replacements = match.get('replacements', [])
        if replacements:
            best_replacement = replacements[0]['value']
            corrected_text = corrected_text[:offset] + best_replacement + corrected_text[offset + length:]
        issue_type = match.get('rule', {}).get('issueType')
        if issue_type == 'misspelling':
            lexical_resource_score -= 0.5
        elif issue_type == 'grammar':
            grammatical_range_and_accuracy_score -= 0.5
        elif issue_type == 'punctuation':
            grammatical_range_and_accuracy_score -= 0.25
        else:
            lexical_resource_score -= 0.25

    task_response_score = max(task_response_score, 6)
    coherence_and_cohesion_score = max(coherence_and_cohesion_score, 6)
    lexical_resource_score = max(lexical_resource_score, 6)
    grammatical_range_and_accuracy_score = max(grammatical_range_and_accuracy_score, 6)
    band_score = (task_response_score + coherence_and_cohesion_score + lexical_resource_score + grammatical_range_and_accuracy_score) / 4

    return corrected_text, band_score

# Streamlit UI Function to display content
def display_writing2_content():
    st.title("IELTS Writing Task Generator")

    # Use Streamlit session state to store random_task
    if 'random_task' not in st.session_state:
        st.session_state.random_task = None

    # When 'Generate Random Writing Task' button is clicked
    if st.button("Generate Random Writing Task"):
        if writing_tasks:
            # Reset session state to clear previous outputs
            st.session_state.random_task = None
            st.session_state.user_answer = ""

            # Generate and display new task
            st.session_state.random_task = random.choice(writing_tasks)
            st.write("Random IELTS Writing Task 2 Question:")
            st.write(st.session_state.random_task['text'])
            
            current_url = ielts_test_urls[writing_tasks.index(st.session_state.random_task)]
            st.write(f"Source URL: {current_url}")

            if st.session_state.random_task['sample_answer']:
                st.write("Sample Answer:")
                st.write(st.session_state.random_task['sample_answer'])
            else:
                st.write("No sample answer available.")
        else:
            st.write("No tasks available to display.")
    
    # User inputs their answer
    user_answer = st.text_area("Enter your answer:", key='user_answer')

    if user_answer:
        corrected_text, band_score, grammar_result = check_grammar_with_languagetool(user_answer)
        
        st.write("Corrected Text:")
        st.write(corrected_text)
        
        st.write("Band Score:")
        st.write(band_score)
        
        if st.session_state.random_task and st.session_state.random_task.get('sample_answer'):
            question = st.session_state.random_task['text']
            sample_answer = st.session_state.random_task['sample_answer']
            feedback = check_answer_correctness(question, user_answer, sample_answer)
            st.write("Answer Correctness Feedback:")
            st.write(feedback)
        else:
            st.write("No sample answer available to check correctness.")

if __name__ == "__main__":
    # Run as a Streamlit app
    # display_writing2_content()

    # Run as a FastAPI app
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)
