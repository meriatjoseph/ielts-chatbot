import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyttsx3

# Load environment variables for OpenAI API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Section Three template
template = '''
Generate a script for an IELTS Listening task for Section Three. 
This section should include a conversation between three speakers discussing an academic or research-based topic. 
Example topics include a tutor and student discussing an assignment or a group of students planning a research project.

After the script, generate 10 *Short Answer* questions, divided by difficulty level as follows:
- 3 Easy questions
- 3 Medium questions
- 4 Hard questions

Present everything in the following format as plain text:

Script:
[Generated conversation between multiple speakers]

Questions (Short Answer):
- Easy Level:
   1. [Easy question]
   2. [Easy question]
   3. [Easy question]

- Medium Level:
   1. [Medium question]
   2. [Medium question]
   3. [Medium question]

- Hard Level:
   1. [Hard question]
   2. [Hard question]
   3. [Hard question]
   4. [Hard question]

Answers:
- Easy Level:
   1. [Answer]
   2. [Answer]
   3. [Answer]

- Medium Level:
   1. [Answer]
   2. [Answer]
   3. [Answer]

- Hard Level:
   1. [Answer]
   2. [Answer]
   3. [Answer]
   4. [Answer]
'''

# Streamlit interface for Section Three
st.title("IELTS Listening Task Generator - Section Three")

# Button to generate the task
if st.button("Generate Listening Task for Section Three"):
    # Initialize LLM and create chain for Section Three
    llm = ChatGroq(temperature=0.7, max_tokens=2000)
    prompt = PromptTemplate(input_variables=[], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate response
    response = chain.run({})
    
    # Display the generated task directly as text
    st.text_area("Generated Listening Task - Section Three", response, height=500)
    
    # Extract the script from the response
    script_start = response.find("Script:") + len("Script:")
    script_end = response.find("Questions (Short Answer):")
    script = response[script_start:script_end].strip()
    
    # Split script into lines assuming each line corresponds to one speaker
    lines = script.split("\n")
    
    # Initialize pyttsx3
    engine = pyttsx3.init()
    
    # Set up voices for three speakers
    voices = engine.getProperty('voices')
    speaker_voices = {
        "Speaker 1": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0",  # Speaker 1: Hazel
        "Speaker 2": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",  # Speaker 2: David
        "Speaker 3": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"   # Speaker 3: Zira
    }
    
    # Determine which speaker each line belongs to and maintain consistent voice mapping
    for line in lines:
        # Assuming each line starts with the speaker name, e.g., "Speaker 1: ..."
        if ": " in line:
            speaker_name, dialogue = line.split(": ", 1)
            voice_token = speaker_voices.get(speaker_name, voices[0].id)  # Default to the first voice if not found
            engine.setProperty('voice', voice_token)
            engine.say(dialogue)  # Queue the speech
            engine.runAndWait()  # Ensure the current speech is finished before moving to the next
    
    # Final cleanup of any remaining queued speech
    engine.runAndWait()