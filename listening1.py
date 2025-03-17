import os
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json  # Use json for safer parsing

# Load environment variables for API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Template for Section Two
template_section_two = '''
Generate a script for an IELTS Listening task for Section Two. 
This should be a monologue by one speaker in an everyday social context. 
Example topics include a speech about student services on a university campus or arrangements for meals during a conference.

After the script, generate 10 **Multiple Choice** questions, divided by difficulty level as follows:
- 3 Easy questions
- 3 Medium questions
- 4 Hard questions

Present everything in the following format as JSON:
{
  "status": "success",
  "script": "[Generated monologue]",
  "questions": {
    "easy": [
      { "question": "[Easy question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Easy question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Easy question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ],
    "medium": [
      { "question": "[Medium question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Medium question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Medium question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ],
    "hard": [
      { "question": "[Hard question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 4]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ]
  }
}
'''

# Template for Section One
template_section_one = '''
Generate a script for an IELTS Listening task for Section One. 
This should be a dialogue between two speakers in an everyday social context. 
Example topics include making a booking for an event or confirming travel arrangements.

After the script, generate 10 **Multiple Choice** questions, divided by difficulty level as follows:
- 3 Easy questions
- 3 Medium questions
- 4 Hard questions

Present everything in the following format as JSON:
{
  "status": "success",
  "script": "[Generated dialogue]",
  "questions": {
    "easy": [
      { "question": "[Easy question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Easy question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Easy question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ],
    "medium": [
      { "question": "[Medium question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Medium question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Medium question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ],
    "hard": [
      { "question": "[Hard question 1]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 2]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 3]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" },
      { "question": "[Hard question 4]", "options": ["A", "B", "C", "D"], "answer": "[Correct Option]" }
    ]
  }
}
'''

def generate_listening_task_section_one():
    """
    Generates an IELTS Listening task for Section One in JSON format.
    """
    try:
        llm = ChatGroq(temperature=0.7, max_tokens=2000)
        prompt = PromptTemplate(input_variables=[], template=template_section_one)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({})
        
        # Use json.loads for safer parsing
        task = json.loads(response)
        return task
    except Exception as e:
        return {"status": "error", "message": str(e)}

def save_script_as_audio(script: str) -> BytesIO:
    """
    Converts a script to audio using gTTS and returns it as a BytesIO stream.
    Omits speaker prefixes if present.
    """
    # Remove any speaker prefixes for cleaner TTS output
    cleaned_script = "\n".join(line.split(": ", 1)[1] if ": " in line else line for line in script.splitlines())
    tts = gTTS(cleaned_script, lang="en")
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    return audio_stream

# Streamlit interface for Section Two
st.title("IELTS Listening Task Generator - Section Two")

# Button to generate the task for Section Two
if st.button("Generate Listening Task for Section Two"):
    try:
        # Initialize LLM and create chain for Section Two
        llm = ChatGroq(temperature=0.7, max_tokens=2000)
        prompt = PromptTemplate(input_variables=[], template=template_section_two)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate response
        response = chain.run({})
        
        # Parse the response into JSON-like format
        task = json.loads(response)  # Use json.loads for safe parsing
        
        # Extract script and questions
        script = task.get("script", "")
        questions = task.get("questions", {})
        
        # Display script and questions in Streamlit
        st.subheader("Generated Script:")
        st.text(script)
        
        st.subheader("Questions:")
        for level, question_list in questions.items():
            st.write(f"### {level.capitalize()} Level:")
            for q in question_list:
                st.write(f"- {q['question']}")
                for option in q['options']:
                    st.write(f"  - {option}")
                st.write(f"*Answer:* {q['answer']}")
        
        # Generate and play audio from script
        st.subheader("Generated Audio:")
        audio_stream = save_script_as_audio(script)
        st.audio(audio_stream, format="audio/mpeg")
    
    except Exception as e:
        st.error(f"Error generating task: {str(e)}")

# Streamlit - Upload custom script for TTS
custom_script = st.text_area("Enter a custom script to generate audio (optional):")
if st.button("Generate Audio from Custom Script"):
    if custom_script.strip():
        try:
            audio_stream = save_script_as_audio(custom_script)
            st.audio(audio_stream, format="audio/mpeg")
        except Exception as e:
            st.error(f"Error generating audio: {str(e)}")
    else:
        st.warning("Please enter a script to generate audio.")
