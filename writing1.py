import os
import io
import fitz  # PyMuPDF
import random
import bs4
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import requests

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPEN_API_KEY')

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", api_key=open_api_key)

# Define the prompt template for generating similar IELTS writing tasks
prompt_template = ChatPromptTemplate.from_template(
    """Based on the content provided, generate a similar IELTS Writing Task 1 prompt.
    Use the context below to create a new, unique task.
    
    <context>
    {context}
    </context>

    Task: Generate a similar IELTS Writing Task 1 prompt."""
)

# Function to extract IELTS Writing Task 1 questions, images, and sample answers from web documents
def extract_writing_tasks_from_web(urls):
    tasks = []
    
    # Loop through each URL and perform web scraping
    for url in urls:
        print(f"Processing URL: {url}")

        try:
            # Make an HTTP GET request to fetch the page content
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            # Parse the HTML content with BeautifulSoup
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            
            # Find all elements with the specified class for text
            elements = soup.find_all("div", class_="et_pb_section et_pb_section_0 et_section_regular")
            print(f"Number of elements found: {len(elements)}")  # Print the number of elements found

            # Loop through each element and extract the text and image
            for element in elements:
                task = element.get_text(strip=True)
                if task:  # Ensure the task is not empty
                    # Attempt to find an associated image
                    img_tag = element.find_next("img")  # Find the first image following the element
                    img_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else None

                    # Find the sample answer
                    sample_answer_div = soup.find("div", class_="et_pb_toggle_content clearfix")
                    sample_answer = sample_answer_div.get_text(strip=True) if sample_answer_div else None

                    # Append the task with text, image URL, and sample answer
                    tasks.append({
                        "text": task,
                        "image_url": img_url,
                        "sample_answer": sample_answer
                    })
                    
                    print(f"Extracted Task: {task}")  # Debugging log
                    if img_url:
                        print(f"Extracted Image URL: {img_url}")  # Debugging log
                    if sample_answer:
                        print(f"Extracted Sample Answer: {sample_answer}")  # Debugging log

        except requests.exceptions.RequestException as e:
            print(f"Error processing {url}: {e}")

    return tasks

# Function to generate IELTS Writing Test URLs with leading zeros
def generate_ielts_test_urls():
    base_url = "https://ieltstrainingonline.com/ielts-writing-practice-test-"
    urls = [f"{base_url}{i:02d}/" for i in range(1, 100)]  
    return urls

# Generate the test URLs
ielts_test_urls = generate_ielts_test_urls()

# Extract writing tasks from web documents
writing_tasks = extract_writing_tasks_from_web(ielts_test_urls)

# Function to create vector embeddings and load documents
def create_vector_embedding():
    embeddings = OpenAIEmbeddings()
    loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
    pdf_docs = loader.load()  # Document loading

    if not pdf_docs:
        st.error("No documents loaded from PDF. Please check the document loader.")
        return

    # Combine documents from web scraping and PDF loading
    docs = pdf_docs + writing_tasks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    if not final_documents:
        st.error("Document splitting failed. Please check the text splitter.")
        return

    if not embeddings:
        st.error("Embedding creation failed. Please check the embedding generator.")
        return

    if final_documents and embeddings:
        vectors = FAISS.from_documents(final_documents, embeddings)
    else:
        st.error("Failed to create vectors. Documents or embeddings are empty.")
        return

    # Extract tasks for RAG-based question generation
    extracted_tasks = extract_writing_tasks(final_documents)  # Use a different variable name to avoid conflicts
    random.shuffle(extracted_tasks)

# Function to extract IELTS Writing Task 1 tasks and images from documents
def extract_writing_tasks(documents):
    tasks = []
    unique_tasks = set()  # Use a set to ensure uniqueness

    for i, doc in enumerate(documents):
        content = doc.page_content
        pdf_path = doc.metadata.get('source', 'Unknown source')
        page_number = doc.metadata.get('page_number', 1)

        # Log to check which document and page is being processed
        print(f"Processing Document: {pdf_path}, Page: {page_number}")

        if "20 minutes" in content:
            text = content.strip()  # Strip whitespace to ensure proper uniqueness checks
            if text not in unique_tasks:  # Check if the task is unique
                unique_tasks.add(text)

                # Log the extracted text and metadata
                print(f"Task {len(tasks)+1} found in document {pdf_path} on page {page_number}")

                # Extract images from the PDF
                images = extract_images_from_pdf(pdf_path, start_page=page_number)
                tasks.append({"text": text, "image": images[0] if images else None})
            else:
                print(f"Duplicate task found in document {pdf_path} on page {page_number} and skipped.")
    
    # Final log for debugging
    print(f"Total tasks extracted: {len(tasks)}")
    return tasks

# Function to extract images from a PDF file, starting from a specific page
def extract_images_from_pdf(pdf_path, start_page=1):
    images = []
    pdf_document = fitz.open(pdf_path)  # Open the PDF document

    num_pages = pdf_document.page_count  # Get the total number of pages

    for page_number in range(start_page - 1, num_pages):  # Start checking from the start_page
        page = pdf_document.load_page(page_number)  # Load the current page
        print(f"Checking document {pdf_path}, Page: {page_number + 1}")

        # Retrieve images on the current page
        image_list = page.get_images(full=True)

        if image_list:  # If there are images on this page
            print(f"Found {len(image_list)} image(s) on page {page_number + 1}")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)

            # Stop searching after the first page with images is found
            break
        else:
            print(f"No images found on page {page_number + 1}")

    if images:
        print(f"Image(s) extracted successfully from document {pdf_path}")
    else:
        print(f"No images found starting from page {start_page} in document {pdf_path}")
    
    return images

# Function to check the correctness of the user's answer using LLM
def check_answer_correctness(user_answer, sample_answer):
    if not user_answer or not sample_answer:
        return "Insufficient data to check correctness."

    # Create messages using SystemMessage and HumanMessage
    messages = [
        SystemMessage(content="You are an expert in evaluating IELTS writing tasks."),
        HumanMessage(content=f"Evaluate the following text based on its content, coherence, structure, and overall quality. "
                             f"Provide feedback on the strengths and weaknesses of the response, focusing on clarity, grammar, and relevance to the task."
                             f"\n\nUser's Answer:\n{user_answer}\n\nFeedback:")
    ]

    # Generate feedback using LLM
    response = llm(messages=messages)  # Adjusting the function call here

    # Extract feedback from the response
    feedback = response.content if response else "Could not generate feedback."

    return feedback


# Function to generate a similar question using RAG approach
def generate_similar_question_with_rag():
    # Ensure vector embeddings are available
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()  # Use the vector store as retriever
        document_chain = create_stuff_documents_chain(llm, prompt_template)  # Create the document chain with LLM and prompt
        retrieval_chain = create_retrieval_chain(retriever, document_chain)  # Create retrieval chain
        
        # Randomly select an initial task to find similar ones
        initial_task = random.choice(st.session_state.writing_tasks)
        response = retrieval_chain.invoke({'input': initial_task['text']})

        print(f"Response generated by RAG: {response['answer']}")  # Debug print

        # Use the RAG-generated question to create an image prompt
        generated_text = response['answer']
        image_prompt = create_image_prompt_from_text(generated_text)
        
        # Generate or retrieve an image based on the prompt
        new_image = generate_image_from_prompt(image_prompt)

        # Return response with the newly generated or retrieved image
        return {"text": generated_text, "image": new_image}
    else:
        # Return a default dictionary if vector embeddings are not ready
        return {"text": "Vector embeddings are not yet ready.", "image": None}

# Function to create an image prompt from text
def create_image_prompt_from_text(text):
    # Extract key information from the text to create a descriptive prompt for image generation
    return f"A chart or diagram illustrating {text.lower()}"

# Function to generate images based on the prompt
def generate_image_from_prompt(prompt_text):
    # Replace with your actual API call to an image generation model like DALL-E or Stable Diffusion
    print(f"Generating image for: {prompt_text}")
    
    # Example: Call an external API or a local model to generate the image
    # Simulating image URL generation (replace with actual API/model call)
    image_url = generate_image_using_model(prompt_text)
    
    # If an image is generated, return it
    if image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        return image

    return None  # Return None if no image is generated

# Function to generate an image using a model
def generate_image_using_model(prompt_text):
    # This function should call an API for image generation (e.g., DALL-E, Stable Diffusion)
    print(f"Generating image from model for: {prompt_text}")
    # Placeholder for API call to an image generation service
    return None  # Replace with the URL of the generated image

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

    # Ensure the scores do not go below 6 (minimum band score for the specified range)
    task_response_score = max(task_response_score, 6)
    coherence_and_cohesion_score = max(coherence_and_cohesion_score, 6)
    lexical_resource_score = max(lexical_resource_score, 6)
    grammatical_range_and_accuracy_score = max(grammatical_range_and_accuracy_score, 6)

    # Calculate the band score as an average of all four components
    band_score = (task_response_score + coherence_and_cohesion_score + lexical_resource_score + grammatical_range_and_accuracy_score) / 4

    return corrected_text, band_score

def display_writing1_content():
    st.title("IELTS Writing Task Generator")

    # Use Streamlit session state to store random_task
    if 'random_task' not in st.session_state:
        st.session_state.random_task = None  # Initialize random_task in session state

    # Randomly select a question and display when button is clicked
    if st.button("Generate Random Writing Task"):
        if writing_tasks:
            st.session_state.random_task = random.choice(writing_tasks)
            st.write("Random IELTS Writing Task 1 Question:")
            st.write(st.session_state.random_task['text'])
            
            # Display the current URL
            current_url = ielts_test_urls[writing_tasks.index(st.session_state.random_task)]
            st.write(f"Source URL: {current_url}")

            # Display associated image if available
            if st.session_state.random_task['image_url']:
                st.image(st.session_state.random_task['image_url'], caption="Associated Image")
            
            # Display the sample answer if available
            if st.session_state.random_task['sample_answer']:
                st.write("Sample Answer:")
                st.write(st.session_state.random_task['sample_answer'])
            else:
                st.write("No sample answer available.")
        else:
            st.write("No tasks available to display.")
    
    # User inputs their answer
    user_answer = st.text_area("Enter your answer:")

    if user_answer:
        # Check grammar using LanguageTool
        corrected_text, band_score, grammar_result = check_grammar_with_languagetool(user_answer)
        
        st.write("Corrected Text:")
        st.write(corrected_text)
        
        st.write("Band Score:")
        st.write(band_score)
        
        # Display grammar suggestions
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

        # Check correctness using LLM if a sample answer is available
        if st.session_state.random_task and st.session_state.random_task.get('sample_answer'):
            feedback = check_answer_correctness(user_answer, st.session_state.random_task['sample_answer'])
            st.write("Answer Correctness Feedback:")
            st.write(feedback)
        else:
            st.write("No sample answer available to check correctness.")

# Run the Streamlit app
if __name__ == "__main__":
    display_writing1_content()
