import os
import random
import requests
import bs4
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPEN_API_KEY')
print(f"Loaded OpenAI API key: {open_api_key}")  # Debug: Check if API key is loaded correctly

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5", api_key=open_api_key, temperature=0.0, max_tokens=3000)
print("LLM initialized with model gpt-3.5.")  # Debug: Confirm that LLM is initialized

# FastAPI instance
api_app = FastAPI()

# Pydantic model for request validation
class ReadingTaskRequest(BaseModel):
    user_answer: str

# Function to extract reading tasks from multiple URLs
def extract_reading_tasks_from_web(urls):
    tasks = []  # This will hold the JSON objects for each URL
    for url in urls:
        try:
            print(f"Processing URL: {url}")  # Debug: Track which URL is being processed
            # Make a request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors
            
            # Parse the page content using BeautifulSoup
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            print(f"Successfully fetched and parsed the URL: {url}")  # Debug: Confirm page fetch success
            
            # Find the specific div that contains the text you want
            target_elements = soup.find_all("div", class_="et_pb_module et_pb_text et_pb_text_0 et_pb_text_align_left et_pb_bg_layout_light")
            print(f"Found {len(target_elements)} 'et_pb_module et_pb_text' div elements in {url}")  # Debug: Check how many target div elements were found
            
            # Extract text from the specific divs
            for element in target_elements:
                # Initialize a list to hold all the passage content
                passages = []
                
                # Find all <p> tags with the exact style
                paragraphs = element.find_all("p", style="text-align: justify;")
                for paragraph in paragraphs:
                    # Now find all <span> elements inside this <p> with the specific style
                    spans = paragraph.find_all("span", style="text-align: justify; font-size: large;")
                    for span in spans:
                        # Extract the text from each span and add to the passages
                        passage_text = span.get_text(strip=True)
                        if passage_text:
                            passages.append(passage_text)

                if passages:
                    # Create the JSON structure
                    task_json = {
                        "passage": passages,  # Add the passages to the "passage" key
                        "questions": []  # Placeholder for questions
                    }
                    tasks.append(task_json)  # Append to tasks list
                    print(f"Extracted JSON: {task_json}")  # Debug: Display the JSON structure
                
        except requests.exceptions.RequestException as e:
            # Handle any request-related errors
            print(f"Error processing {url}: {e}")  # Debug: Log the error
    
    print(f"Total tasks extracted: {len(tasks)}")  # Debug: Confirm how many tasks were extracted in total
    return tasks  # Return the list of tasks in JSON format

# Function to generate IELTS Reading Test URLs with leading zeros
def generate_ielts_reading_urls():
    base_url = "https://ieltstrainingonline.com/ielts-reading-practice-test-"
    urls = [f"{base_url}{i:02d}-with-answers/" for i in range(8, 20)]  
    print(f"Generated URLs: {urls}")  # Debug: Output the generated URLs
    return urls

# Streamlit UI Function to display content
def display_reading1_content():
    st.title("IELTS Reading Task Generator")

    # When 'Generate Random Reading Task' button is clicked
    if st.button("Generate Random Reading Task"):
        # Regenerate tasks from URLs on button click
        ielts_reading_urls = generate_ielts_reading_urls()  # Generate fresh URLs
        reading_tasks = extract_reading_tasks_from_web(ielts_reading_urls)  # Extract tasks from URLs

        if reading_tasks:
            # Select a random task
            random_task = random.choice(reading_tasks)
            
            st.write("Random IELTS Reading Task Content:")
            st.write(random_task)  # Display the extracted task content
        else:
            st.write("No tasks available to display.")
    
if __name__ == "__main__":
    # Run as a Streamlit app
    display_reading1_content()

    # Run as a FastAPI app
    # import uvicorn
    # uvicorn.run(api_app, host="0.0.0.0", port=8000)
