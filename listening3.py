import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyttsx3
import tempfile

# Load environment variables for OpenAI API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Template for Section Three Listening Task
template = '''
Generate a script for an IELTS Listening task for Section Three. 
This section should include a conversation between three speakers discussing an academic or research-based topic. 
Example topics include a tutor and student discussing an assignment or a group of students planning a research project.

After the script, generate 10 Short Answer questions, divided by difficulty level as follows:
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

def generate_listening_task_section_three():
    llm = ChatGroq(temperature=0.7, max_tokens=4000)
    prompt = PromptTemplate(input_variables=[], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({})

    script_start = response.find("Script:") + len("Script:")
    script_end = response.find("Questions (Short Answer):")
    questions_start = response.find("Questions (Short Answer):") + len("Questions (Short Answer):")
    answers_start = response.find("Answers:")

    script = response[script_start:script_end].strip()
    questions_text = response[questions_start:answers_start].strip()
    answers_text = response[answers_start + len("Answers:"):].strip()

    questions = {"Easy Level": [], "Medium Level": [], "Hard Level": []}
    answers = {"Easy Level": [], "Medium Level": [], "Hard Level": []}

    for level in questions:
        level_start = questions_text.find(f"- {level}:")
        if level_start != -1:
            level_end = questions_text.find("-", level_start + 1)
            level_text = questions_text[level_start:level_end].strip() if level_end != -1 else questions_text[level_start:].strip()
            questions[level] = [q.strip() for q in level_text.split("\n")[1:]]

    for level in answers:
        level_start = answers_text.find(f"- {level}:")
        if level_start != -1:
            level_end = answers_text.find("-", level_start + 1)
            level_text = answers_text[level_start:level_end].strip() if level_end != -1 else answers_text[level_start:].strip()
            answers[level] = [a.strip() for a in level_text.split("\n")[1:]]

    return {"script": script, "questions": questions, "answers": answers}

def save_script_as_audio(script):
    lines = script.split("\n")
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    speaker_voices = {
        "Speaker 1": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0",
        "Speaker 2": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
        "Speaker 3": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        for line in lines:
            if ": " in line:
                speaker_name, dialogue = line.split(": ", 1)
                voice_token = speaker_voices.get(speaker_name, voices[0].id)
                engine.setProperty("voice", voice_token)
                engine.say(dialogue)
                engine.save_to_file(dialogue, temp_audio_file.name)
        return temp_audio_file.name
