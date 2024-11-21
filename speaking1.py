import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv('OPEN_API_KEY')

# Function to load and parse the provided PDF to extract topics and questions
def extract_topics_and_questions():
    st.write("Extracting topics and questions from the PDF...")

    if "topics_with_questions" not in st.session_state:
        loader = PyPDFLoader("speaking1_pdf/Speaking_Part_1_Questions.pdf")
        st.session_state.docs = loader.load()

        if not st.session_state.docs:
            st.write("No documents found! Please check your PDF.")
            return

        full_text = "\n".join(doc.page_content for doc in st.session_state.docs)
        st.session_state.topics_with_questions = process_topics_and_questions(full_text)
        st.session_state.final_documents = st.session_state.docs
        st.write("Topics and questions extracted successfully.")

# Function to process and return topics with questions
def process_topics_and_questions(pdf_content):
    topics_with_questions = {}
    current_topic = None
    lines = pdf_content.split("\n")

    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            # Treat lines starting with a number (e.g., 1. Topic Name) as a new topic
            current_topic = line
            topics_with_questions[current_topic] = []
        elif current_topic and line.startswith("•"):
            # Treat bullet points under a topic as questions
            topics_with_questions[current_topic].append(line)
    
    return topics_with_questions

# Function to create vector embeddings for RAG based on PDF content
def create_vector_embedding_for_rag():
    if "vectors" not in st.session_state:
        if "final_documents" in st.session_state:
            try:
                st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.write("FAISS vector store initialized successfully.")
            except Exception as e:
                st.write(f"Error initializing FAISS vector store: {e}")
        else:
            st.write("No documents available to create embeddings.")

# Function to generate new unique topics and questions using RAG
def generate_new_topics_and_questions_using_rag():
    language_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()

        # Retrieve examples from the PDF
        retrieved_context_docs = retriever.get_relevant_documents("IELTS Speaking Part 1 Topics")
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_context_docs)

        # Debugging: Display retrieved context
        st.write("Retrieved Context for RAG Generation:")
        st.write(retrieved_context)

        # Prompt to generate unique topics and questions
        prompt_for_new_topics = (
            "Based on the retrieved examples, generate entirely new IELTS Speaking Part 1 topics. "
            "Each topic should have 5 unique and creative questions suitable for conversational practice. "
            "Categorize the questions properly under their topics. Avoid duplicating examples."
        )

        final_prompt = f"{prompt_for_new_topics}\n\nRetrieved Examples:\n{retrieved_context}\n\nGenerated Topics and Questions:"
        
        try:
            generated_topics_response = language_model.invoke(final_prompt)
            generated_content = getattr(generated_topics_response, 'content', str(generated_topics_response))
            
            # Debugging: Display generated content
            st.write("Generated Content from RAG:")
            st.write(generated_content)

            # Process the generated content into topics and questions
            extra_topics_with_questions = process_generated_topics_and_questions(generated_content)
            return extra_topics_with_questions
        except Exception as e:
            st.write(f"Error generating new topics and questions: {e}")
            return None
    else:
        st.write("Vector store not initialized for RAG.")
        return None

# Function to parse generated topics and questions
def process_generated_topics_and_questions(content):
    extra_topics_with_questions = {}
    current_topic = None

    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            # Treat lines starting with a number as a new topic
            current_topic = line
            extra_topics_with_questions[current_topic] = []
        elif current_topic and line.startswith("•"):
            # Append questions under the current topic
            extra_topics_with_questions[current_topic].append(line)
    
    return extra_topics_with_questions

# Function to merge original and generated topics/questions
def merge_topics_and_questions():
    all_topics_with_questions = st.session_state.topics_with_questions.copy()

    if "extra_topics_with_questions" in st.session_state:
        all_topics_with_questions.update(st.session_state.extra_topics_with_questions)

    st.session_state.all_topics_with_questions = all_topics_with_questions
    st.write("Merged topics and questions successfully.")

# Display topics and their respective questions
def display_topics_and_questions():
    if "all_topics_with_questions" in st.session_state:
        selected_topic = st.selectbox("Select a Topic", list(st.session_state.all_topics_with_questions.keys()))
        if selected_topic:
            st.write(f"**Selected Topic: {selected_topic}**")
            questions = st.session_state.all_topics_with_questions[selected_topic]
            if questions:
                st.write("### Questions:")
                for idx, question in enumerate(questions, start=1):
                    st.write(f"{idx}. {question.strip()}")
            else:
                st.write("No questions found for the selected topic.")
    else:
        st.write("No topics available.")

# Initialize the app
if "topics_with_questions" not in st.session_state:
    extract_topics_and_questions()
    create_vector_embedding_for_rag()

    # Generate RAG-based topics and questions
    extra_topics_with_questions = generate_new_topics_and_questions_using_rag()
    if extra_topics_with_questions:
        st.session_state.extra_topics_with_questions = extra_topics_with_questions
    else:
        st.write("No new topics generated using RAG.")

    merge_topics_and_questions()

# Display the dropdown and questions
display_topics_and_questions()
