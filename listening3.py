from typing import Dict, Any
from gtts import gTTS
import tempfile
import os

def generate_listening_task_section_three() -> Dict[str, Any]:
    """
    Generates a listening task for IELTS Section Three with a conversation script
    and related questions divided into difficulty levels.
    """
    # Example script
    script = """Speaker 1: Hi, Professor, I wanted to discuss my research topic for the assignment.
Speaker 2: Of course. Have you decided on the area of focus?
Speaker 1: Yes, I’m thinking of analyzing the impact of social media on mental health.
Speaker 3: That’s an interesting choice. Are you focusing on any particular demographic?
Speaker 1: Primarily young adults, aged 18 to 25."""

    # Example questions and answers
    questions_answers = {
        "easy": [
            {"question": "What is the topic of the student's research?", "answer": "The impact of social media on mental health."},
            {"question": "Who is the main demographic focus?", "answer": "Young adults aged 18 to 25."},
            {"question": "Who is the professor in the conversation?", "answer": "Speaker 2."},
        ],
        "medium": [
            {"question": "What age group is being focused on in the research?", "answer": "18 to 25 years old."},
            {"question": "What is the relationship between Speaker 1 and Speaker 2?", "answer": "Student and professor."},
            {"question": "What kind of topic does Speaker 3 find interesting?", "answer": "The impact of social media on mental health."},
        ],
        "hard": [
            {"question": "What specific aspect of mental health might be explored?", "answer": "The impact of social media on mental health for young adults."},
            {"question": "Why does Speaker 1 choose the demographic of 18 to 25?", "answer": "They are the primary users of social media."},
            {"question": "What potential challenges could arise in the research?", "answer": "Difficulties in isolating social media as the primary factor for mental health issues."},
            {"question": "What does Speaker 2 suggest about the topic?", "answer": "This is left open for interpretation in the script."},
        ],
    }

    # Assemble the output
    return {
        "status": "success",
        "script": script,
        "questions": {
            "easy": [qa["question"] for qa in questions_answers["easy"]],
            "medium": [qa["question"] for qa in questions_answers["medium"]],
            "hard": [qa["question"] for qa in questions_answers["hard"]],
        },
        "answers": {
            "easy": [qa["answer"] for qa in questions_answers["easy"]],
            "medium": [qa["answer"] for qa in questions_answers["medium"]],
            "hard": [qa["answer"] for qa in questions_answers["hard"]],
        },
    }


def save_script_as_audio(script: str) -> str:
    """
    Converts the given script into an audio file and saves it.
    """
    try:
        # Use gTTS for text-to-speech conversion
        tts = gTTS(script, lang="en")
        
        # Create a temporary file for saving the audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "ielts_section_three.mp3")
        
        # Save the audio file
        tts.save(audio_path)
        
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate audio: {e}")
