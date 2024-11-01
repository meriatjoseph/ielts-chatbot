import os
import random
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables for OpenAI API key and Groq API key
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Function to load and parse the provided PDF to extract topics and questions
def extract_topics_and_questions():
    st.write("Extracting topics and questions from the PDF...")

    if "topics_with_questions" not in st.session_state:
        loader = PyPDFLoader("speaking1_pdf/Speaking_Part_1_Questions.pdf")
        st.session_state.docs = loader.load()
        st.session_state.final_documents = st.session_state.docs

        if not st.session_state.docs:
            st.write("No documents found! Please check your PDF.")
            return

        full_text = "\n".join(doc.page_content for doc in st.session_state.docs)
        st.session_state.topics_with_questions = process_topics_and_questions(full_text)
        st.write("Topics and questions extracted successfully.")

# Function to process and return topics with questions
def process_topics_and_questions(pdf_content):
    topics_with_questions = {}
    current_topic = None
    lines = pdf_content.split("\n")

    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            current_topic = line
            topics_with_questions[current_topic] = []
        elif current_topic and line.startswith("•"):
            topics_with_questions[current_topic].append(line)
    
    return topics_with_questions

# Function to create vector embeddings for RAG based on PDF content
def create_vector_embedding_for_rag():
    if "vectors" not in st.session_state:
        if "final_documents" in st.session_state and st.session_state.final_documents:
            try:
                st.session_state.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPEN_API_KEY'))
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.write("FAISS vector store initialized successfully.")
            except Exception as e:
                st.write(f"Error initializing FAISS vector store: {e}")
        else:
            st.write("No documents available to create embeddings.")

# Function to generate new unique topics and their questions using RAG with PDF as context
def generate_new_topics_and_questions_using_rag():
    language_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

    if "topics_with_questions" in st.session_state and "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        
        # Generate topics based on retrieved examples from the PDF
        prompt_for_new_topics = (
            "Using the following retrieved examples as context, create entirely new and unique IELTS Speaking Part 1 topics. "
            "Each topic should be different from those retrieved, with 5 related questions. Ensure the topics are suitable for conversational practice."
        )
        
        # Retrieve relevant content from the PDF as examples
        example_topic = "IELTS Speaking Part 1 Topics"
        retrieved_context_docs = retriever.get_relevant_documents(example_topic)
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_context_docs)

        # Add the retrieved context to the prompt
        final_prompt = f"{prompt_for_new_topics}\n\nRetrieved Examples:\n{retrieved_context}\n\nNew Topics and Questions:"

        try:
            # Call the language model with the prompt
            generated_topics_response = language_model.invoke(final_prompt)
            
            # Check if response has content
            if hasattr(generated_topics_response, 'content'):
                generated_content = generated_topics_response.content
            else:
                generated_content = str(generated_topics_response)
                
            # Process the response to extract topics and questions
            extra_topics_with_questions = process_generated_topics_and_questions(generated_content)
            
        except Exception as e:
            st.write(f"Error in generating new topics and questions from ChatGroq: {e}")
            return None

        return extra_topics_with_questions
    else:
        return None

# Helper function to parse the generated topics and questions
def process_generated_topics_and_questions(content):
    extra_topics_with_questions = {}
    current_topic = None

    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        # Detect topic headings
        if line and line[0].isdigit() and "." in line:
            current_topic = line
            extra_topics_with_questions[current_topic] = []
        elif current_topic and line.startswith("•"):
            extra_topics_with_questions[current_topic].append(line)
    
    return extra_topics_with_questions

# Function to merge the original topics and questions with the generated ones
def merge_and_shuffle_questions():
    if "extra_topics_with_questions" in st.session_state:
        all_topics_with_questions = st.session_state.topics_with_questions.copy()
        all_topics_with_questions.update(st.session_state.extra_topics_with_questions)
        st.session_state.all_topics_with_questions = all_topics_with_questions

        for topic, questions in all_topics_with_questions.items():
            random.shuffle(questions)

        st.write("Merged and shuffled topics and questions successfully.")
    else:
        st.session_state.all_topics_with_questions = st.session_state.topics_with_questions

# Initialize the app with topics and questions from PDF and RAG
if "topics_with_questions" not in st.session_state:
    extract_topics_and_questions()
    create_vector_embedding_for_rag()

    extra_topics_with_questions = generate_new_topics_and_questions_using_rag()
    if extra_topics_with_questions:
        st.session_state.extra_topics_with_questions = extra_topics_with_questions
        st.session_state.show_generated_extra = True
    else:
        st.write("Failed to generate extra topics and questions.")

    merge_and_shuffle_questions()

# Display the merged and shuffled topics and questions
st.write("Final Merged Topics and Questions:")
if "all_topics_with_questions" in st.session_state:
    selected_topic = st.selectbox("Select a Topic", list(st.session_state.all_topics_with_questions.keys()))
    
    if st.button("Generate Random Question"):
        merged_questions = st.session_state.all_topics_with_questions[selected_topic]
        if merged_questions:
            selected_question = random.choice(merged_questions)
            st.write(f"**Topic: {selected_topic}**")
            st.write(f"**Random Question:** {selected_question.strip()}")
        else:
            st.write("No questions found for the selected topic.")
else:
    st.write("No topics found.")
