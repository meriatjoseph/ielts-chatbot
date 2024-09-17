import os
import random
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import requests
import bs4  # Ensure BeautifulSoup is imported

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPEN_API_KEY')

# Initialize LLM with Chat Model
llm = ChatOpenAI(model="gpt-4", api_key=open_api_key)

# Function to extract IELTS Writing Task 2 questions and sample answers from web documents
def extract_writing_tasks_from_web(urls):
    tasks = []
    for url in urls:
        print(f"Processing URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            
            # Find the elements for Task 2 questions
            elements = soup.find_all("div", class_="et_pb_section et_pb_section_1 et_section_regular")
            print(f"Number of Task 2 question elements found: {len(elements)}")
            
            for element in elements:
                task = element.get_text(strip=True)
                if task:
                    # Find the parent div for the sample answer
                    sample_answer_parent_div = soup.find("div", class_="et_pb_module et_pb_toggle et_pb_toggle_1 et_pb_toggle_item et_pb_toggle_close")
                    # Find the specific div within the parent
                    sample_answer_div = sample_answer_parent_div.find("div", class_="et_pb_toggle_content clearfix") if sample_answer_parent_div else None
                    
                    # Extract all text content within the div, including p, span, and other nested tags
                    if sample_answer_div:
                        sample_answer = sample_answer_div.get_text(separator=' ', strip=True)
                    else:
                        sample_answer = None
                    
                    # Append the extracted data to tasks
                    tasks.append({
                        "text": task,
                        "sample_answer": sample_answer,
                        "url": url  # Add the current URL to the task data
                    })
        except requests.exceptions.RequestException as e:
            print(f"Error processing {url}: {e}")
    return tasks

# Function to generate IELTS Writing Test URLs with leading zeros
def generate_ielts_test_urls():
    base_url = "https://ieltstrainingonline.com/ielts-writing-practice-test-"
    urls = [f"{base_url}{i:02d}/" for i in range(1, 11)]  
    return urls

# Generate the test URLs
ielts_test_urls = generate_ielts_test_urls()

# Extract writing tasks from web documents for Writing Task 2
writing_tasks = extract_writing_tasks_from_web(ielts_test_urls)

# Streamlit UI Function to display content
def display_writing2_content():
    st.title("IELTS Writing Task Generator")

    # Ensure that writing tasks have been extracted
    if not writing_tasks:
        st.write("No writing tasks found. Please check the URL or extraction logic.")
        return

    # Initialize session state for question generation
    if 'question_generated' not in st.session_state:
        st.session_state.question_generated = False

    # Function to update question on button click
    def update_question():
        # Randomly select a task from the extracted tasks
        selected_task = random.choice(writing_tasks)
        st.session_state.current_task = selected_task
        st.session_state.question_generated = True

    # Button to generate a random Writing Task 2 question
    if st.button("Generate Random Writing Task"):
        update_question()

    # Display the randomly generated question and sample answer
    if st.session_state.question_generated:
        st.write("Random IELTS Writing Task 2 Question:")
        st.write(st.session_state.current_task["text"])

        # Display the URL associated with the question
        st.write("Source URL:")
        st.write(st.session_state.current_task["url"])

        if st.session_state.current_task["sample_answer"]:
            st.write("Sample Answer:")
            st.write(st.session_state.current_task["sample_answer"])
        else:
            st.write("No sample answer available for this task.")

