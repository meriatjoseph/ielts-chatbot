import os
import random
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Function to load and parse the provided PDF to extract topics and questions
def extract_topics_and_questions():
    st.write("Extracting topics and questions from the PDF...")

    if "topics_with_questions" not in st.session_state:
        # Load the provided PDF using PyPDFLoader
        loader = PyPDFLoader("speaking1_pdf/Speaking Part 1 Questions.pdf")

        # Load the documents and store them in session state
        st.session_state.docs = loader.load()
        st.session_state.final_documents = st.session_state.docs  # Save the documents here

        if not st.session_state.docs:
            st.write("No documents found! Please check your PDF.")
            return

        # Combine all page content into a single string
        full_text = ""
        for doc in st.session_state.docs:
            full_text += doc.page_content + "\n"

        # Process and extract topics and questions from the combined content
        st.session_state.topics_with_questions = process_topics_and_questions(full_text)
        st.write("Topics and questions extracted successfully.")

# Function to process and return topics with questions
def process_topics_and_questions(pdf_content):
    """Extract topics and questions from the document."""
    topics_with_questions = {}
    current_topic = None

    # Split the content by lines
    lines = pdf_content.split("\n")

    for line in lines:
        line = line.strip()
        # Detect topic headings (based on numbering and topic titles)
        if line and line[0].isdigit() and "." in line:  # Assuming topic numbering, e.g., '1. Desserts/ Cakes'
            current_topic = line
            topics_with_questions[current_topic] = []
        # Detect questions starting with bullet points "•"
        elif current_topic and line.startswith("•"):
            topics_with_questions[current_topic].append(line)

    return topics_with_questions

# Function to generate similar extra topics and their questions using RAG
def generate_extra_topics_and_questions_using_rag():
    language_model = ChatOpenAI(api_key=os.getenv('OPEN_API_KEY'), model="gpt-4")
    
    # Generate extra topics based on existing topics
    if "topics_with_questions" in st.session_state:
        existing_topics = list(st.session_state.topics_with_questions.keys())
        prompt_for_topics = f"Based on the following topics, generate similar IELTS Speaking Part 1 topics:\n" + "\n".join(existing_topics)
        
        # Get the response from the language model
        generated_topics_response = language_model(prompt_for_topics)
        
        # Extract topics from the response, assuming they are in a content field or separated by newlines
        generated_topics = []
        if hasattr(generated_topics_response, 'content'):
            # If the response has a 'content' attribute (like a JSON-like object), extract topics
            content_str = generated_topics_response.content
            generated_topics = content_str.strip().split('\n')
        else:
            # Otherwise, assume the response is a simple string with topics separated by newlines
            generated_topics = str(generated_topics_response).strip().split('\n')

        extra_topics_with_questions = {}

        # Generate questions for each generated topic
        for topic in generated_topics:
            topic = topic.strip()
            if topic:  # Ensure non-empty topic
                retriever = st.session_state.vectors.as_retriever()

                # Create a prompt to generate questions for the new topic
                prompt_for_questions = f"Generate 5 questions related to the topic '{topic}'."
                
                rag_chain = RetrievalQA.from_chain_type(
                    llm=language_model,
                    chain_type="stuff",
                    retriever=retriever
                )

                generated_questions_response = rag_chain.run(prompt_for_questions)
                questions = str(generated_questions_response).strip().split('\n')
                
                # Store the generated topic and questions
                extra_topics_with_questions[topic] = questions

        return extra_topics_with_questions

    else:
        return None

# Function to generate questions using RAG for an existing topic
def generate_questions_using_rag(selected_topic):
    language_model = ChatOpenAI(api_key=os.getenv('OPEN_API_KEY'))  # Ensure language_model is defined here
    if "topics_with_questions" in st.session_state:
        topics = st.session_state.all_topics_with_questions
        if topics and selected_topic in topics:
            questions = topics[selected_topic]

            # Initialize RAG (FAISS vector search)
            retriever = st.session_state.vectors.as_retriever()

            # Retrieve relevant content
            retrieved_content = retriever.get_relevant_documents(selected_topic)

            # Create a prompt for generating questions
            prompt = f"Based on the following topic '{selected_topic}' and retrieved context, generate new questions:\n"
            prompt += "\n".join(questions) + "\n\nRetrieved Context:\n" + "\n".join(doc.page_content for doc in retrieved_content)

            # Create a chain for RAG
            rag_chain = RetrievalQA.from_chain_type(
                llm=language_model,
                chain_type="stuff",
                retriever=retriever
            )

            # Use RAG to generate new questions
            generated_questions_response = rag_chain.run(prompt)
            generated_questions = str(generated_questions_response).strip().split('\n')  # Split generated questions into a list

            # Merge original and generated questions and shuffle
            all_questions = questions + generated_questions
            random.shuffle(all_questions)  # Shuffle the questions

            return all_questions  # Return the merged, shuffled questions
        else:
            return None
    else:
        return None

# Function to merge the original topics and questions with the generated ones
def merge_and_shuffle_questions():
    # Merge existing topics and questions with generated ones
    if "extra_topics_with_questions" in st.session_state:
        all_topics_with_questions = st.session_state.topics_with_questions.copy()  # Start with original topics
        all_topics_with_questions.update(st.session_state.extra_topics_with_questions)  # Add generated topics
        st.session_state.all_topics_with_questions = all_topics_with_questions

        # Shuffle questions within each topic
        for topic, questions in all_topics_with_questions.items():
            random.shuffle(questions)  # Shuffle the questions within each topic

        st.write("Merged and shuffled topics and questions successfully.")
    else:
        st.session_state.all_topics_with_questions = st.session_state.topics_with_questions

# Initialize vector store and embeddings for RAG
def create_vector_embedding_for_rag():
    if "vectors" not in st.session_state:
        if "final_documents" in st.session_state:  # Ensure final_documents is initialized
            st.session_state.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPEN_API_KEY'))
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        else:
            st.write("No documents available to create embeddings.")

# Initialize the app with topics and questions from PDF and RAG
if "topics_with_questions" not in st.session_state:
    extract_topics_and_questions()

    # Initialize FAISS vector database for RAG
    create_vector_embedding_for_rag()

    # Generate extra topics and questions using RAG on init
    extra_topics_with_questions = generate_extra_topics_and_questions_using_rag()
    if extra_topics_with_questions:
        st.session_state.extra_topics_with_questions = extra_topics_with_questions
        st.session_state.show_generated_extra = True  # Flag to show generated extra topics and questions
    else:
        st.write("Failed to generate extra topics and questions.")

    # Merge and shuffle the questions
    merge_and_shuffle_questions()

# Display the merged and shuffled topics and questions
st.write("Final Merged Topics and Questions:")
if "all_topics_with_questions" in st.session_state:
    # Dropdown to select a topic
    selected_topic = st.selectbox("Select a Topic", list(st.session_state.all_topics_with_questions.keys()))
    
    # Button to generate a random question for the selected topic
    if st.button("Generate Random Question"):
        merged_questions = generate_questions_using_rag(selected_topic)
        if merged_questions:
            # Pick a random question from the list
            selected_question = random.choice(merged_questions)
            st.write(f"**Topic: {selected_topic}**")
            st.write(f"**Random Question:** {selected_question.strip()}")
        else:
            st.write("No questions found for the selected topic.")
else:
    st.write("No topics found.")
