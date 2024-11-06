import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

# Load environment variables for OpenAI API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define prompt template for a single passage with multiple question types
passage_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on a unique academic topic.
Choose a topic related to advanced scientific research, innovative technology, or an emerging field in natural or physical sciences, social sciences, or global issues.
Example topics include recent breakthroughs in renewable energy, the psychology of learning, or climate change adaptation.

After the passage, create exactly:
1. 5 "True/False" questions with answers as either "True" or "False".
2. 5 multiple-choice questions, each with 4 answer choices labeled A, B, C, and D. Ensure each question has 4 distinct options and one correct answer labeled as the corresponding letter.
3. 5 "Complete the Sentence" questions with 4 answer choices labeled A, B, C, and D, with one correct answer labeled as the corresponding letter.

Ensure all questions, options, and answers are generated correctly. Format the questions and answers in JSON, where each question includes:
- "question_text" (string)
- "options" (for multiple-choice and complete-the-sentence questions, each option should be a list of strings)
- "answer" (correct answer as a single string or letter)

Output JSON with "title", "passage", and a "questions" list. Each question set includes:
- "question_type" (string)
- "questions" (list of questions)
- "options" (list of options for each question type that requires it)
- "answers" (list of correct answers as actual text, not labels).'''

# Initialize LLM and create chain for the passage
llm = ChatGroq(temperature=0.7, max_tokens=2000)
prompt = PromptTemplate(input_variables=[], template=passage_template)
chain = LLMChain(llm=llm, prompt=prompt)

def validate_json_structure(parsed_json):
    """
    Validates the JSON structure to ensure it contains the required 'title', 'passage',
    and 'questions' sections.
    """
    try:
        if not isinstance(parsed_json, dict):
            print("Validation failed: Root element is not a dictionary.")
            return False

        # Check for title, passage, and questions keys
        if "title" not in parsed_json or "passage" not in parsed_json or "questions" not in parsed_json:
            print("Validation failed: Missing 'title', 'passage', or 'questions' in JSON structure.")
            return False

        # Validate questions structure
        questions = parsed_json["questions"]
        if not isinstance(questions, list):
            print("Validation failed: 'questions' is not a list.")
            return False
        for section in questions:
            if "question_type" not in section or "questions" not in section or "answers" not in section:
                print(f"Validation failed: Question section missing keys - {section}")
                return False
            if section["question_type"] in ["Multiple Choice", "Complete the Sentence"] and "options" not in section:
                print("Validation failed: 'options' missing in a question that requires it.")
                return False

        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def parse_response(response):
    """
    Parses the response text into separate components: title, passage, and questions/answers.
    Maps answer labels to actual text for 'MultipleChoice' and 'CompleteTheSentence' question types.
    Returns empty strings or empty lists for missing components.
    """
    try:
        # Match title, passage content, and questions JSON section
        title_match = re.search(r'"title": "(.*?)",', response)
        passage_match = re.search(r'"passage": "(.*?)",', response, re.DOTALL)
        question_json_match = re.search(r'"questions": \[.*\]', response, re.DOTALL)

        # Extract title, passage, and questions data, defaulting to empty values if missing
        title = title_match.group(1).strip() if title_match else ""
        passage_text = passage_match.group(1).strip() if passage_match else ""
        
        questions_data = []
        if question_json_match:
            try:
                questions_data = json.loads("{" + question_json_match.group(0) + "}")["questions"]
            except json.JSONDecodeError:
                print("Failed to parse 'questions' as JSON.")

        # Restructure questions for each type with empty lists if data is missing
        questions = []
        for section in questions_data:
            q_type = section.get("question_type", "")
            if q_type == "True/False":
                questions.append({
                    "question_type": q_type,
                    "questions": [q.get("question_text", "") for q in section.get("questions", [])],
                    "answers": [a.get("answer", "False") for a in section.get("answers", [])]
                })
            elif q_type == "Multiple Choice":
                options = [opt.get("options", ["A", "B", "C", "D"]) for opt in section.get("options", [])]
                answers = [
                    options[i][ord(answer) - ord("A")]  # Convert label to actual text
                    for i, answer in enumerate([a.get("answer", "A") for a in section.get("answers", [])])
                ]
                questions.append({
                    "question_type": q_type,
                    "questions": [q.get("question_text", "") for q in section.get("questions", [])],
                    "options": options,
                    "answers": answers
                })
            elif q_type == "Complete the Sentence":
                options = [opt.get("options", ["A", "B", "C", "D"]) for opt in section.get("options", [])]
                answers = [
                    options[i][ord(answer) - ord("A")]  # Convert label to actual text
                    for i, answer in enumerate([a.get("answer", "A") for a in section.get("answers", [])])
                ]
                questions.append({
                    "question_type": q_type,
                    "questions": [q.get("question_text", "") for q in section.get("questions", [])],
                    "options": options,
                    "answers": answers
                })

        return title, passage_text, questions

    except Exception as e:
        print(f"Error parsing response: {e}")
        return "", "", []

def generate_reading_test_json():
    """
    Generates the JSON for an IELTS Academic Reading passage with three types of questions.
    """
    try:
        # Generate response for the passage
        response = chain.run({})

        # Debug output to examine the raw response from the model
        print("Raw response:", response)

        # Attempt to load response as JSON directly if possible
        try:
            parsed_json = json.loads(response)
            if validate_json_structure(parsed_json):
                print("Direct JSON parsing successful.")
                return {"status": "success", **parsed_json}
            else:
                print("Direct JSON parsing failed validation; proceeding with regex parsing.")
        except json.JSONDecodeError:
            print("Raw response is not valid JSON; proceeding with regex parsing.")

        # Parse the response into title, passage, and questions
        title, passage, questions = parse_response(response)

        # If any part is missing, return with status "partial"
        if not title or not passage or not questions:
            return {
                "status": "partial",
                "title": title or "",
                "passage": passage or "",
                "questions": questions or []
            }

        # Form the JSON output with separated title, passage, and questions
        ielts_test_json = {
            "status": "success",
            "title": title,
            "passage": passage,
            "questions": questions
        }
        return ielts_test_json

    except Exception as e:
        # Return a structured error response if generation fails
        print(f"Error generating reading test JSON: {e}")
        return {"status": "error", "detail": f"Error generating reading test JSON: {e}"}
