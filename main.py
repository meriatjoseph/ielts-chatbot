from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from writing1 import generate_random_task as generate_writing1_task, check_grammar_with_languagetool, check_answer_correctness as check_answer_correctness1
from writing2 import generate_random_task as generate_writing2_task, check_answer_correctness as check_answer_correctness2
from speaking1 import get_random_topic_with_questions, generate_feedback_for_answer  # Import functions from speaking1
import random

app = FastAPI()

# Pydantic models for request bodies
class WritingTaskRequest(BaseModel):
    user_answer: str

class WritingTaskResponse(BaseModel):
    question: str
    sample_answer: str
    image_url: str = None  # Add image_url as an optional field

class SpeakingTaskRequest(BaseModel):
    user_answer: str

class SpeakingTaskResponse(BaseModel):
    topic: str
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the IELTS Task API"}

@app.get("/writing1/generate_task/", response_model=WritingTaskResponse)
def generate_task_writing1():
    try:
        task = generate_writing1_task()
        sample_answer = task.get('sample_answer', 'No sample answer available')
        image_url = task.get('image_url', None)  # Get the image URL
        return {
            "question": task['text'],
            "sample_answer": sample_answer,
            "image_url": image_url  # Include the image URL in the response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/writing1/check_grammar/")
def check_grammar_writing1(request: WritingTaskRequest):
    try:
        corrected_text, band_score, grammar_result = check_grammar_with_languagetool(request.user_answer)
        return {
            "corrected_text": corrected_text,
            "band_score": band_score,
            "grammar_result": grammar_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/writing1/evaluate_answer/")
def evaluate_answer_writing1(request: WritingTaskRequest):
    try:
        task = generate_writing1_task()
        feedback = check_answer_correctness1(task['text'], request.user_answer, task['sample_answer'])
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/writing2/generate_task/", response_model=WritingTaskResponse)
def generate_task_writing2():
    try:
        task = generate_writing2_task()
        sample_answer = task.get('sample_answer', 'No sample answer available')
        return {"question": task['text'], "sample_answer": sample_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/writing2/check_grammar/")
def check_grammar_writing2(request: WritingTaskRequest):  # Correct function name
    try:
        corrected_text, band_score, grammar_result = check_grammar_with_languagetool(request.user_answer)
        return {
            "corrected_text": corrected_text,
            "band_score": band_score,
            "grammar_result": grammar_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/writing2/evaluate_answer/")
def evaluate_answer_writing2(request: WritingTaskRequest):
    try:
        task = generate_writing2_task()
        feedback = check_answer_correctness2(task['text'], request.user_answer, task['sample_answer'])
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speaking1/generate_task/", response_model=SpeakingTaskResponse)
def generate_task_speaking1():
    try:
        topic, questions = get_random_topic_with_questions()
        question = random.choice(questions)
        return {
            "topic": topic,
            "question": question
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speaking1/evaluate_answer/")
def evaluate_answer_speaking1(request: SpeakingTaskRequest):
    try:
        topic, questions = get_random_topic_with_questions()
        question = random.choice(questions)
        feedback = generate_feedback_for_answer(topic, question, request.user_answer)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn main:app --reload
