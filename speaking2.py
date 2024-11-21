import os
import random
import streamlit as st
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
import openai
import tempfile
from dotenv import load_dotenv
from io import BytesIO
from gtts import gTTS
from playsound import playsound
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables for API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Function to perform TTS using gTTS and return audio stream
def text_to_speech_stream(text: str) -> BytesIO:
    try:
        # Use gTTS to convert text to speech
        tts = gTTS(text=text, lang="en")
        
        # Save the audio to a BytesIO stream
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        return audio_stream

    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {e}")
        return None

# Function to save audio to a file
def save_audio(audio_stream: BytesIO, file_name: str = "corrected_audio.mp3"):
    with open(file_name, "wb") as f:
        f.write(audio_stream.read())
    return file_name

# Function to calculate cosine similarity between two texts using embeddings
def calculate_similarity(text1, text2):
    embeddings = OpenAIEmbeddings()
    embedding1 = embeddings.embed_query(text1)
    embedding2 = embeddings.embed_query(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# Function to load and parse the PDF for Speaking Part 2
def create_vector_embedding_for_speaking_part2():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("speaking2_pdf")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.session_state.speaking2_tasks = extract_speaking2_questions(st.session_state.final_documents)

# Function to extract questions from the document
def extract_speaking2_questions(documents):
    tasks = []
    for doc in documents:
        content = doc.page_content
        if not content:
            continue
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)
                if len(parts) == 2:
                    question = parts[1].strip()
                    tasks.append({"question": question})
    return tasks

# Function to generate similar questions using RAG
def generate_similar_questions_using_rag():
    if "speaking2_tasks" in st.session_state and st.session_state.speaking2_tasks:
        tasks = st.session_state.speaking2_tasks
        random_task = random.choice(tasks)
        question = random_task['question']
        retriever = st.session_state.vectors.as_retriever()
        language_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        prompt = f"Based on the following question, generate similar questions:\n{question}"
        rag_chain = RetrievalQA.from_chain_type(
            llm=language_model,
            chain_type="stuff",
            retriever=retriever
        )
        rag_chain.run(prompt)
        return question
    else:
        return None

# Function to transcribe uploaded audio using OpenAI Whisper
def transcribe_audio(file):
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(file.read())
            temp_audio_path = temp_audio_file.name
        try:
            with open(temp_audio_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
            os.remove(temp_audio_path)
            
            # Debugging log
            print(f"Transcription Result: {transcription}")

            return transcription['text']
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    return None

# Initialize the language model
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Function to check the correctness of the user's answer using LLM and similarity score
def check_answer_correctness(user_answer, generated_question):
    if not user_answer or not generated_question:
        print("Debug: Missing user_answer or generated_question")
        return "Insufficient data to check correctness.", "Match feedback not available.", "Correction not available.", False

    print(f"Debug: User Answer: {user_answer}")
    print(f"Debug: Generated Question: {generated_question}")

    messages = [
        SystemMessage(content="You are an expert in evaluating IELTS speaking tasks."),
        HumanMessage(content=f"Evaluate the following user's response to an IELTS speaking question. "
                             f"Analyze the response for sentence structure, grammar, vocabulary usage, and coherence with the question. "
                             f"Additionally, assess how well the response addresses the generated question.\n\n"
                             f"Generated Question:\n{generated_question}\n\n"
                             f"User's Response:\n{user_answer}\n\n"
                             f"Provide detailed feedback, question-user match feedback, and start the corrected version with 'Corrected Version:'.")
    ]
    try:
        response = llm(messages=messages)
        
        if response:
            content = response.content
            feedback = content.split("Question-User Match Feedback:")[0].strip()
            match_feedback_section = content.split("Question-User Match Feedback:")[1] if "Question-User Match Feedback:" in content else None
            corrected_text_section = content.split("Corrected Version:")[1] if "Corrected Version:" in content else None

            match_feedback = match_feedback_section.strip() if match_feedback_section else "Match feedback not available."
            corrected_text = corrected_text_section.strip() if corrected_text_section else "Correction not available."

            similarity_score = calculate_similarity(user_answer, generated_question)
            print(f"Debug: Similarity Score: {similarity_score}")
            status = similarity_score > 0.75  # True if similarity score is above threshold
            return feedback, match_feedback, corrected_text, status
        else:
            return "Could not generate feedback.", "Match feedback not available.", "Correction not available.", False
    except Exception as e:
        print(f"Error in check_answer_correctness: {e}")
        return "Error occurred during evaluation.", "Match feedback not available.", "Correction not available.", False

# Function to display Speaking Part 2 content
def display_speaking2_content():
    st.title("IELTS Speaking Part 2 Question Generator with Feedback")

    if "speaking2_tasks" not in st.session_state:
        create_vector_embedding_for_speaking_part2()

    if st.button("Generate Random Speaking Part 2 Question"):
        original_question = generate_similar_questions_using_rag()
        if original_question:
            st.session_state.original_question = original_question
        else:
            st.write("No questions found or tasks are not yet ready.")

    if 'original_question' in st.session_state:
        st.write(f"Question: {st.session_state.original_question}")
        audio_file = st.file_uploader("Upload your audio response (WAV format)", type=["wav"])

        if audio_file is not None:
            transcription = transcribe_audio(audio_file)
            if transcription:
                st.write(f"*Transcription:* {transcription}")
                
                st.write("*Feedback and Analysis:*")
                general_feedback, match_feedback, corrected_text, status = check_answer_correctness(transcription, st.session_state.original_question)
                st.write(f"*General Feedback:* {general_feedback}")
                st.write(f"*Question-User Match Feedback:* {match_feedback}")
                st.write(f"*Status:* {'True' if status else 'False'}")
                st.write(f"*Corrected Text:* {corrected_text}")
                
                if corrected_text and corrected_text != "Correction not available.":
                    audio_stream = text_to_speech_stream(corrected_text)
                    if audio_stream:
                        audio_file_path = save_audio(audio_stream)
                        if st.button("Play Corrected Text Audio"):
                            playsound(audio_file_path)
                
            else:
                st.write("Failed to transcribe the audio. Please try again.")

# Run the app
if __name__ == "__main__":
    display_speaking2_content()
