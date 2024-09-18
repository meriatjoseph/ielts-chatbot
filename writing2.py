import os
import random
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage  # Import SystemMessage for LLM evaluation
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
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
    urls = [f"{base_url}{i:02d}/" for i in range(1, 11)]  
    return urls

# Generate the test URLs
ielts_test_urls = generate_ielts_test_urls()

# Extract writing tasks from web documents for Writing Task 2
writing_tasks = extract_writing_tasks_from_web(ielts_test_urls)

# Create Stuff Documents Function (for RAG)
def create_stuff_documents(tasks):
    documents = []
    for task in tasks:
        doc_content = f"Task: {task['text']}\nSample Answer: {task['sample_answer'] or 'No sample answer available'}"
        documents.append(Document(page_content=doc_content, metadata={"url": task["url"]}))
    
    return documents

# Create documents for the RAG process
stuff_documents = create_stuff_documents(writing_tasks)

# Initialize the FAISS vector store with the documents created
def create_faiss_retriever(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    task_documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=open_api_key)

    vector_store = FAISS.from_documents(task_documents, embeddings)
    return vector_store.as_retriever()

# Create a FAISS retriever from the documents
retriever = create_faiss_retriever(stuff_documents)

# Create a RAG chain using the retriever and LLM
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

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

# Function to update the question using RAG and generate a sample answer
def update_question():
    # Clear previous question, sample answer, and user answer from session state
    st.session_state.current_task = None
    st.session_state.user_answer = ""
    st.session_state.feedback = ""

    # Use RAG to generate a similar IELTS Writing Task 2 question
    if retriever:
        query = "Generate a similar IELTS Writing Task 2 question based on the following prompt:\n" + random.choice(writing_tasks)['text']
        rag_result = qa_chain.run(query)
        st.session_state.current_task = {
            'text': rag_result,
            'sample_answer': None  # Since it's a generated question, we now generate the answer
        }
        st.session_state.question_generated = True
        
        # Now generate a sample answer using the LLM for the generated question
        messages = [
            SystemMessage(content="You are an expert in generating sample answers for IELTS Writing Task 2."),
            HumanMessage(content=f"Generate a high-quality sample answer for the following IELTS Writing Task 2 question:\n\n{rag_result}")
        ]
        response = llm(messages=messages)
        generated_sample_answer = response.content if response else "Could not generate a sample answer."
        
        # Store the generated sample answer
        st.session_state.current_task['sample_answer'] = generated_sample_answer

# Streamlit UI Function to display content
def display_writing2_content():
    st.title("IELTS Writing Task Generator")

    if not writing_tasks:
        st.write("No writing tasks found. Please check the URL or extraction logic.")
        return

    if 'question_generated' not in st.session_state:
        st.session_state.question_generated = False

    if st.button("Generate Random Writing Task"):
        update_question()

    if st.session_state.question_generated:
        st.write("Random IELTS Writing Task 2 Question:")
        st.write(st.session_state.current_task['text'])

        sample_answer = st.session_state.current_task.get('sample_answer')
        st.write(f"Sample Answer: {sample_answer or 'No sample answer available.'}")

    # Input area for user answer
    user_answer = st.text_area("Enter your answer:", key='user_answer')

    if user_answer:
        if st.session_state.question_generated:
            question = st.session_state.current_task['text']
            sample_answer = st.session_state.current_task['sample_answer']
            feedback = check_answer_correctness(question, user_answer, sample_answer)
            st.write("Answer Correctness Feedback:")
            st.write(feedback)
            
            # Check grammar with LanguageTool and display the result
            corrected_text, band_score, grammar_result = check_grammar_with_languagetool(user_answer)
            st.write("Corrected Text from LanguageTool:")
            st.write(corrected_text)
            st.write(f"Estimated Band Score: {band_score}")
        else:
            st.write("Please generate a question first.")

if __name__ == '__main__':
    display_writing2_content()
