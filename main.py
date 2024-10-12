import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from writing1 import generate_random_task as generate_writing1_task, check_grammar_with_languagetool, check_answer_correctness as check_answer_correctness1
from writing2 import generate_random_task as generate_writing2_task, check_answer_correctness as check_answer_correctness2
from speaking2 import create_vector_embedding_for_speaking_part2, generate_similar_questions_using_rag as generate_speaking2_question
# from vocabulary_grammar import generate_vocabulary_task as generate_vocabulary_task,generate_grammar_task as generate_grammar_task
# from grammar import generate_grammar_task_from_grammer as generate_grammar_task_from_grammer
import random
import uvicorn
import asyncio
import streamlit as st

# FastAPI app instance
api_app = FastAPI()
app = FastAPI()
# Pydantic models for request bodies
class WritingTaskRequest(BaseModel):
    user_answer: str

class WritingTaskResponse(BaseModel):
    question: str
    sample_answer: str
    image_url: Optional[str] = None  # Add image_url as an optional field

class VocabularyTaskResponse(BaseModel):
    vocabulary_task: str
    correct_answers: str

class GrammarTaskJsonResponse(BaseModel):
    gaps: dict
    answers: dict
    text: str

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
def check_grammar_writing2(request: WritingTaskRequest):
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

# @app.get("/vocabulary/generate_task/", response_model=VocabularyTaskResponse)
# def generate_vocabulary_tasks():
#     try:
#         task = generate_vocabulary_task()  # Ensure this function returns a dict with the task and answers
#         vocabulary_task = task['vocabulary_task']  # Modify according to your function's response structure
#         correct_answers = task['correct_answers']  # Modify according to your function's response structure
#         return {
#             "vocabulary_task": vocabulary_task,
#             "correct_answers": correct_answers
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.get("/grammar/generate_task_with_json")
# def generate_grammar_task_with_json():
#     try:
#         # Call your existing grammar task generation function
#         task_result = generate_grammar_task_from_grammer()

#         # Log the raw task_result for debugging purposes
#         logging.info("Raw task result: %s", task_result)

#         # Return the raw task_result as it is
#         return task_result

#     except ValueError as ve:
#         logging.error(f"ValueError: {str(ve)}")
#         raise HTTPException(status_code=500, detail=str(ve))
#     except Exception as e:
#         logging.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/speaking2/generate_question/")
def generate_speaking2_task():
    try:
        create_vector_embedding_for_speaking_part2()
        question = generate_speaking2_question()
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if loop.is_running():
        config = uvicorn.Config(api_app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        loop.create_task(server.serve())
    else:
        uvicorn.run(api_app, host="0.0.0.0", port=8000)


# Run the app with: uvicorn main:app --reload  (do not remove this comment)