import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Load environment variables for OpenAI API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define prompt templates for different passages and question types

# Passage 1: TRUE/FALSE/NOT GIVEN + Short Answer questions
passage1_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on the topic of {topic}.
After the passage, create 5 TRUE/FALSE/NOT GIVEN questions and 4 Short Answer questions.
Make sure the passage and questions are suitable for the IELTS exam.'''

# Passage 2: Multiple Choice + Summary Completion
passage2_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on the topic of {topic}.
After the passage, create 5 Multiple Choice questions and 5 Summary Completion questions.
Make sure the passage and questions are suitable for the IELTS exam.'''

# Passage 3: Matching Headings + Sentence Completion
passage3_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on the topic of {topic}.
After the passage, create 5 Matching Headings questions and 5 Sentence Completion questions.
Make sure the passage and questions are suitable for the IELTS exam.'''

# Define prompts for each passage
prompt1 = PromptTemplate(input_variables=["topic"], template=passage1_template)
prompt2 = PromptTemplate(input_variables=["topic"], template=passage2_template)
prompt3 = PromptTemplate(input_variables=["topic"], template=passage3_template)

# Initialize the LLM (OpenAI)
llm = ChatGroq(temperature=0.7, max_tokens=2000)

# Create chains for each passage
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
chain3 = LLMChain(llm=llm, prompt=prompt3)

# Streamlit interface
st.title("IELTS Academic Reading Task Generator")

# User input for topics
passage1_topic = st.text_input("Enter the topic for the first passage (TRUE/FALSE/NOT GIVEN + Short Answer):")
passage2_topic = st.text_input("Enter the topic for the second passage (Multiple Choice + Summary Completion):")
passage3_topic = st.text_input("Enter the topic for the third passage (Matching Headings + Sentence Completion):")

if st.button("Generate IELTS Reading Test"):
    with st.spinner("Generating IELTS Reading Test... This may take a few minutes."):
        
        # Generate first passage with questions
        response1 = chain1.run({"topic": passage1_topic})
        
        # Generate second passage with questions
        response2 = chain2.run({"topic": passage2_topic})
        
        # Generate third passage with questions
        response3 = chain3.run({"topic": passage3_topic})

        # Combine all responses into one
        full_test = f"IELTS Academic Reading Test\n\nREADING PASSAGE 1:\n{response1}\n\nREADING PASSAGE 2:\n{response2}\n\nREADING PASSAGE 3:\n{response3}"

        # Display the generated IELTS Reading Test
        st.subheader("Generated IELTS Academic Reading Test")
        st.markdown(full_test)
        
        # Option to download the test as a text file
        st.download_button(
            label="Download IELTS Reading Test",
            data=full_test,
            file_name="IELTS_Reading_Test.txt",
            mime="text/plain"
        )
