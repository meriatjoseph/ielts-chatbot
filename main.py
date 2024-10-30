import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from writing1 import generate_random_task as generate_writing1_task, check_grammar_with_languagetool, check_answer_correctness as check_answer_correctness1
from writing2 import generate_random_task as generate_writing2_task, check_answer_correctness as check_answer_correctness2
from speaking2 import create_vector_embedding_for_speaking_part2, generate_similar_questions_using_rag as generate_speaking2_question
from reading1 import generate_reading_test_json
from vocabulary import sentence_completion_task, error_correction_task, multiple_choice_task, synonyms_antonyms_task, collocations_task, word_forms_task, context_clues_task, idioms_phrases_task, phrasal_verbs_task

import uvicorn
import asyncio

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

class GrammarTaskJsonResponse(BaseModel):
    gaps: dict
    answers: dict
    text: str

class VocabularyTaskRequest(BaseModel):
    task_type: str = Field(..., description="Type of vocabulary task, e.g., Sentence Completion, Error Correction")

@app.get("/")
def read_root():
    return {"message": "Welcome to the IELTS Task API"}

@app.get("/reading/generated_json")
def get_generated_reading_json():
    """Endpoint to retrieve the JSON for the generated IELTS Reading Test."""
    ielts_test_json = generate_reading_test_json()  # Call the function directly
    if ielts_test_json:
        return ielts_test_json
    else:
        raise HTTPException(status_code=500, detail="Error generating reading test JSON")

@app.get("/writing1/generate_task/", response_model=WritingTaskResponse)
def generate_task_writing1():
    try:
        task = generate_writing1_task()
        sample_answer = task.get('sample_answer', 'No sample answer available')
        image_url = task.get('image_url', None)
        return {
            "question": task['text'],
            "sample_answer": sample_answer,
            "image_url": image_url
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

@app.get("/speaking2/generate_question/")
def generate_speaking2_task():
    try:
        create_vector_embedding_for_speaking_part2()
        question = generate_speaking2_question()
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Vocabulary Task Endpoints
def format_vocabulary_response(task_data):
    """Helper function to format questions and answers to ensure they are strings."""
    questions = [
        question.get("sentence", str(question)) if isinstance(question, dict) else str(question)
        for question in task_data.get("questions", [])
    ]
    correct_answers = [
        answer.get("correct", str(answer)) if isinstance(answer, dict) else str(answer)
        for answer in task_data.get("answers", [])
    ]
    return {
        "vocabulary_task": task_data.get("task", "No task description available"),
        "questions": questions,
        "correct_answers": correct_answers
    }

@app.get("/vocabulary/sentence_completion/")
def generate_sentence_completion_task():
    try:
        task_data = sentence_completion_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/error_correction/")
def generate_error_correction_task():
    try:
        task_data = error_correction_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/multiple_choice/")
def generate_multiple_choice_task():
    try:
        task_data = multiple_choice_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/synonyms_antonyms/")
def generate_synonyms_antonyms_task():
    try:
        task_data = synonyms_antonyms_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/collocations/")
def generate_collocations_task():
    try:
        task_data = collocations_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/word_forms/")
def generate_word_forms_task():
    try:
        task_data = word_forms_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/context_clues/")
def generate_context_clues_task():
    try:
        task_data = context_clues_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/idioms_phrases/")
def generate_idioms_phrases_task():
    try:
        task_data = idioms_phrases_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/phrasal_verbs/")
def generate_phrasal_verbs_task():
    try:
        task_data = phrasal_verbs_task()
        return JSONResponse(content=format_vocabulary_response(task_data))
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

# Run the app with: uvicorn main:app --reload (do not remove this comment) 
