from io import BytesIO
import json
import logging
import tempfile
from fastapi import Form, UploadFile, File
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from writing1 import generate_random_task as generate_writing1_task, check_grammar_with_languagetool, check_answer_correctness as check_answer_correctness1
from writing2 import generate_random_task as generate_writing2_task, check_answer_correctness as check_answer_correctness2
from speaking2 import create_vector_embedding_for_speaking_part2, generate_similar_questions_using_rag as generate_speaking2_question
from reading1 import generate_reading_test_json
from vocabulary import sentence_completion_task, error_correction_task, multiple_choice_task, synonyms_antonyms_task, collocations_task, word_forms_task, context_clues_task, idioms_phrases_task, phrasal_verbs_task
from grammar import (
    past_time_task,
    future_time_task,
    articles_quantifiers_task,
    conditionals_task,
    comparatives_superlatives_task,
    modals_task,
    passive_causative_task,
    compound_future_task,
    quantity_task,
    passive_structures_task,
    uses_of_it_task,
    relative_clauses_task,
    modals_speculation_task,
    talking_about_ability_task,
    emphatic_forms_task,
    wh_words_task
)
from speaking2 import (
    create_vector_embedding_for_speaking_part2,
    generate_similar_questions_using_rag,
    check_answer_correctness,
    transcribe_audio,
)

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
    
# Pydantic models for request bodies and responses
class SpeakingTaskResponse(BaseModel):
    question: str

class AudioCheckRequest(BaseModel):
    user_audio: bytes
    generated_question: str

class AudioCheckResponse(BaseModel):
    general_feedback: str
    match_feedback: str
    corrected_text: str
    status: bool

# class VocabularyTaskRequest(BaseModel):
#     task_type: str = Field(..., description="Type of vocabulary task, e.g., Sentence Completion, Error Correction")

@app.get("/")
def read_root():
    return {"message": "Welcome to the IELTS Task API"}

@app.get("/reading/generated_json")
def get_generated_reading_json():
    """Endpoint to retrieve the JSON for the generated IELTS Reading Test."""
    ielts_test_json = generate_reading_test_json()  # Call the function directly

    # Check the status in the returned JSON and respond accordingly
    if ielts_test_json["status"] == "success":
        return JSONResponse(content=ielts_test_json, status_code=200)
    else:
        # Return a 422 status code for unprocessable entity if parsing failed
        raise HTTPException(status_code=422, detail=ielts_test_json["detail"])
    
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

# @app.get("/speaking2/generate_question/")
# def generate_speaking2_task():
#     try:
#         create_vector_embedding_for_speaking_part2()
#         question = generate_speaking2_question()
#         return {"question": question}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Vocabulary Task Endpoints
@app.get("/vocabulary/sentence_completion/")
def generate_sentence_completion_task():
    try:
        task_data = sentence_completion_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/error_correction/")
def generate_error_correction_task():
    try:
        task_data = error_correction_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/multiple_choice/")
def generate_multiple_choice_task():
    try:
        task_data = multiple_choice_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/synonyms_antonyms/")
def generate_synonyms_antonyms_task():
    try:
        task_data = synonyms_antonyms_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/collocations/")
def generate_collocations_task():
    try:
        task_data = collocations_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/word_forms/")
def generate_word_forms_task():
    try:
        task_data = word_forms_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/context_clues/")
def generate_context_clues_task():
    try:
        task_data = context_clues_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/idioms_phrases/")
def generate_idioms_phrases_task():
    try:
        task_data = idioms_phrases_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vocabulary/phrasal_verbs/")
def generate_phrasal_verbs_task():
    try:
        task_data = phrasal_verbs_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Grammar Task Endpoints
@app.get("/grammar/past_time/")
def generate_past_time_task():
    try:
        task_data = past_time_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/future_time/")
def generate_future_time_task():
    try:
        task_data = future_time_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/articles_quantifiers/")
def generate_articles_quantifiers_task():
    try:
        task_data = articles_quantifiers_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/conditionals/")
def generate_conditionals_task():
    try:
        task_data = conditionals_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/comparatives_superlatives/")
def generate_comparatives_superlatives_task():
    try:
        task_data = comparatives_superlatives_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/modals/")
def generate_modals_task():
    try:
        task_data = modals_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/passive_causative/")
def generate_passive_causative_task():
    try:
        task_data = passive_causative_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/compound_future/")
def generate_compound_future_task():
    try:
        task_data = compound_future_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/quantity/")
def generate_quantity_task():
    try:
        task_data = quantity_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/passive_structures/")
def generate_passive_structures_task():
    try:
        task_data = passive_structures_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/uses_of_it/")
def generate_uses_of_it_task():
    try:
        task_data = uses_of_it_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/relative_clauses/")
def generate_relative_clauses_task():
    try:
        task_data = relative_clauses_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/modals_speculation/")
def generate_modals_speculation_task():
    try:
        task_data = modals_speculation_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/talking_about_ability/")
def generate_talking_about_ability_task():
    try:
        task_data = talking_about_ability_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/emphatic_forms/")
def generate_emphatic_forms_task():
    try:
        task_data = emphatic_forms_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grammar/wh_words/")
def generate_wh_words_task():
    try:
        task_data = wh_words_task()
        return JSONResponse(content=task_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#speaking endpoints
@app.get("/speaking2/generate_question/", response_model=SpeakingTaskResponse)
def generate_speaking2_task():
    try:
        create_vector_embedding_for_speaking_part2()  # Ensure embeddings are loaded
        question = generate_similar_questions_using_rag()
        if question:
            return {"question": question}
        else:
            raise HTTPException(status_code=404, detail="No questions found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speaking2/check_audio/", response_model=AudioCheckResponse)
async def check_audio_response(
    file: UploadFile = File(...),  # File parameter
    question: str = Form(...)      # Form parameter
):
    try:
        # Debugging log
        print(f"Received Question: {question}")

        # Validate file type
        if file.content_type != "audio/mpeg":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an MP3 file.")
        
        # Read the audio file
        audio_content = await file.read()
        
        # Transcribe the audio
        transcription = transcribe_audio(BytesIO(audio_content))
        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe the audio.")

        # Debugging transcription and question
        print(f"Transcription: {transcription}")
        print(f"Generated Question: {question}")

        # Evaluate the transcription
        feedback, match_feedback, corrected_text, status = check_answer_correctness(transcription, question)

        # Log evaluation results
        print(f"Feedback: {feedback}")
        print(f"Match Feedback: {match_feedback}")
        print(f"Corrected Text: {corrected_text}")
        print(f"Status: {status}")

        return {
            "general_feedback": feedback,
            "match_feedback": match_feedback,
            "corrected_text": corrected_text,
            "status": status
        }
    except Exception as e:
        print(f"Error in check_audio_response: {e}")
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
