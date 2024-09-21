from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from writing_tasks import generate_random_task, generate_sample_answer, retrieve_task_by_id
from grammar_check import check_grammar_with_languagetool
from evaluation import check_answer_correctness

app = FastAPI()

# Pydantic models for request bodies
class WritingTaskRequest(BaseModel):
    user_answer: str

class WritingTaskResponse(BaseModel):
    question: str
    sample_answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the IELTS Writing Task API"}

@app.get("/generate_task/", response_model=WritingTaskResponse)
def generate_task():
    try:
        task = generate_random_task()
        sample_answer = generate_sample_answer(task['text'])
        return {"question": task['text'], "sample_answer": sample_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_grammar/")
def check_grammar(request: WritingTaskRequest):
    try:
        corrected_text, band_score, grammar_result = check_grammar_with_languagetool(request.user_answer)
        return {
            "corrected_text": corrected_text,
            "band_score": band_score,
            "grammar_result": grammar_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_answer/")
def evaluate_answer(request: WritingTaskRequest):
    try:
        task = retrieve_task_by_id(request.task_id)
        feedback = check_answer_correctness(task['text'], request.user_answer, task['sample_answer'])
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn main:app --reload
