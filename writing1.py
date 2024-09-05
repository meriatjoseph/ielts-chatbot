import os
import io
import fitz  # PyMuPDF
import random
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PIL import Image
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Define the prompt template for generating IELTS writing tasks
prompt = ChatPromptTemplate.from_template(
    """Based on the content provided, generate a random IELTS Writing Task 1 prompt.
    Do not include any answers or explanations, just the task prompt.
    
    <context>
    {context}
    </context>

    Task: Generate an IELTS Writing Task 1 prompt."""
)

# Function to create vector embeddings and load documents
def create_vector_embedding():
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
    st.session_state.docs = st.session_state.loader.load()  # Document loading

    if not st.session_state.docs:
        st.error("No documents loaded. Please check the document loader.")
        return

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    if not st.session_state.final_documents:
        st.error("Document splitting failed. Please check the text splitter.")
        return

    if not st.session_state.embeddings:
        st.error("Embedding creation failed. Please check the embedding generator.")
        return

    if st.session_state.final_documents and st.session_state.embeddings:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    else:
        st.error("Failed to create vectors. Documents or embeddings are empty.")
        return

    # Proceed with the rest of the function
    st.session_state.writing_tasks = extract_writing_tasks(st.session_state.final_documents)
    random.shuffle(st.session_state.writing_tasks)
    st.session_state.used_tasks = set()

# Function to extract IELTS Writing Task 1 tasks and images from documents
def extract_writing_tasks(documents):
    tasks = []
    unique_tasks = set()  # Use a set to ensure uniqueness

    for i, doc in enumerate(documents):
        content = doc.page_content
        pdf_path = doc.metadata.get('source')
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

# Function to generate a unique random question using the LLM
def generate_llm_question():
    if "vectors" in st.session_state:
        # Filter out tasks that have already been used
        available_tasks = [task for task in st.session_state.writing_tasks if task['text'] not in st.session_state.used_tasks]

        if available_tasks:
            task = random.choice(available_tasks)
            st.session_state.used_tasks.add(task['text'])
            
            # If all tasks have been used, reset the tracker
            if len(st.session_state.used_tasks) == len(st.session_state.writing_tasks):
                st.session_state.used_tasks.clear()
                print("All tasks have been used. Resetting the used tasks tracker.")

            return task
        else:
            return {"text": "No available unique tasks found.", "image": None}
    else:
        return {"text": "Vector embeddings are not yet ready.", "image": None}

# Display function to be called from app.py
def display_writing1_content():
    st.title("IELTS Writing Task Generator")
    
    # Initialize embeddings and vector database on app start
    if "vectors" not in st.session_state:
        create_vector_embedding()

    # Display a single random Task 1 question and update only on button click
    if 'question_generated' not in st.session_state:
        st.session_state.question_generated = False

    def update_question():
        st.session_state.current_task = generate_llm_question()
        st.session_state.question_generated = True

    if st.button("Generate Random Writing Task"):
        update_question()

    if st.session_state.question_generated:
        st.write("Random IELTS Writing Task 1 Question:")
        st.write(st.session_state.current_task.get('text', 'No text available'))

        # Display the associated image if available
        img = st.session_state.current_task.get('image')
        if img:
            st.image(img, caption="Associated Chart Image")
        else:
            st.write("No image available")

    # Allow users to input custom queries
    user_prompt = st.text_input("Enter your query:")

    if user_prompt:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------')
