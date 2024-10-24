# import streamlit as st
# import time
# from google.api_core.exceptions import ResourceExhausted  # Import the exception

# def retry_api_call(api_function, retries=3, delay=60):
#     """Utility function to retry API calls in case of ResourceExhausted error."""
#     for attempt in range(retries):
#         try:
#             return api_function()
#         except ResourceExhausted as e:
#             st.warning(f"Quota exceeded, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
#             time.sleep(delay)
#     st.error("API quota exceeded. Please try again later.")
#     return "Quota exceeded."

# # Lazy import of GoogleGenerativeAIEmbeddings when needed
# def generate_vocabulary_task():
#     """Generate a vocabulary practice task for IELTS General along with the correct answers."""
#     from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import to avoid circular import issues
#     client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
#     return retry_api_call(
#         lambda: client.invoke(
#             "Generate a vocabulary task focused on IELTS General, including words frequently used in IELTS and relevant writing exercises. Also, provide the correct answers for the task. Please return the result in JSON format."
#         ).content
#     )

# def generate_feedback(task_text):
#     """Provide feedback for a vocabulary task response."""
#     from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import to avoid circular import issues
#     client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
#     return retry_api_call(
#         lambda: client.invoke(
#             f"Provide detailed feedback on the following IELTS General vocabulary practice task:\n{task_text}. Please return the result in JSON format."
#         ).content
#     )

# def generate_score(task_text):
#     """Generate a score based on the vocabulary task response."""
#     from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import to avoid circular import issues
#     client = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
#     return retry_api_call(
#         lambda: client.invoke(
#             f"Evaluate the quality of the response for the following IELTS General task and provide a score from 1 to 10. Please return the result in JSON format:\n{task_text}"
#         ).content
#     )

# # Display Vocabulary function
# def display_vocabulary():
#     """Display the vocabulary task interface."""
#     st.title("IELTS Vocabulary Practice Task")

#     # Initialize session state for vocabulary task
#     if 'vocabulary_task' not in st.session_state:
#         st.session_state.vocabulary_task = generate_vocabulary_task()

#     # Process the vocabulary task and extract the answers if present
#     if "Correct Answers:" in st.session_state.vocabulary_task:
#         vocab_task, vocab_answers = st.session_state.vocabulary_task.split("Correct Answers:")
#     else:
#         vocab_task = st.session_state.vocabulary_task
#         vocab_answers = "Correct answers not provided."

#     st.subheader("Vocabulary Task")
#     st.write(vocab_task)

#     # Display correct answers in green color
#     st.subheader("Correct Answers")
#     st.markdown(f"<span style='color:green;'>{vocab_answers.strip()}</span>", unsafe_allow_html=True)

#     # Input for vocabulary task response
#     vocab_task_text = st.text_area("Enter your response to the vocabulary task here:")

#     # Buttons for feedback and score
#     if st.button("Get Vocabulary Feedback and Score"):
#         if vocab_task_text:
#             vocab_feedback = generate_feedback(vocab_task_text)
#             vocab_score = generate_score(vocab_task_text)
#             st.subheader("Vocabulary Feedback")
#             st.write(vocab_feedback)
#             st.subheader("Vocabulary Score")
#             st.write(vocab_score)
#         else:
#             st.warning("Please enter a response to receive feedback and score.")

#     # Button for next vocabulary task
#     if st.button("Next Vocabulary Task"):
#         st.session_state.vocabulary_task = generate_vocabulary_task()
#         st.rerun()  # Refresh the page for the new vocabulary task
