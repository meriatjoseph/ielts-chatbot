import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import re

# Load environment variables for OpenAI API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define prompt templates for each passage
passage1_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on a unique academic topic.
Choose a topic related to advanced scientific research, innovative technology, or an emerging field in the natural or physical sciences.
Example topics include recent breakthroughs in renewable energy, artificial intelligence in healthcare, quantum physics, biochemistry, genetic engineering, astrobiology, or environmental science.
Ensure this topic is unique to this passage and distinct from the other passages in this task.

After the passage, create exactly 5 "Complete the Sentence" questions with options, each with 4 answer choices labeled A, B, C, and D. Provide the correct answer and format the questions and answers in JSON, where each question includes "question_text", "options", and "answer".'''

passage2_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on a unique academic topic in the social sciences, humanities, or the arts.
Possible topics include significant historical transformations, influential sociological theories, cross-cultural studies, psychological theories, economic policies, linguistics, and philosophy.
Example topics include the impact of industrialization, the development of languages, the role of art in society, the psychology of learning, ethics in technology, or the influence of ancient civilizations on modern society.
Ensure this topic is completely distinct from any other passages generated in this task.

After the passage, create exactly 5 "True/False" questions. Provide the correct answer and format the questions and answers in JSON, where each question includes "question_text" and "answer".'''

passage3_template = '''Generate an IELTS Academic Reading passage of about 700-900 words on a unique academic topic in environmental science, global issues, or geographical studies.
Example topics include biodiversity conservation, climate change adaptation, the impact of urbanization on ecosystems, global health challenges, international environmental policies, or the effects of deforestation.
Make sure this topic is entirely distinct from other passage topics in this task.

After the passage, create exactly 5 multiple-choice questions with 4 answer choices labeled A, B, C, and D. Provide the correct answer and format the questions and answers in JSON, where each question includes "question_text", "options", and "answer".'''

# Initialize LLM and create chains for each passage
llm = ChatGroq(temperature=0.7, max_tokens=2000)
prompt1 = PromptTemplate(input_variables=[], template=passage1_template)
prompt2 = PromptTemplate(input_variables=[], template=passage2_template)
prompt3 = PromptTemplate(input_variables=[], template=passage3_template)
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
chain3 = LLMChain(llm=llm, prompt=prompt3)

def parse_response(response):
    """
    Parses the response text into separate components: passage, questions, and answers.
    """
    # Split passage and questions/answers section
    passage_match = re.search(r'^(.*?)\n\nQuestions:', response, re.DOTALL)
    questions_section = response[passage_match.end():] if passage_match else ""

    # Extract passage text
    passage_text = passage_match.group(1).strip() if passage_match else ""

    # Parse questions and answers
    questions = []

    question_json_match = re.search(r'\[.*\]', questions_section, re.DOTALL)
    if question_json_match:
        questions = json.loads(question_json_match.group(0))

    return passage_text, questions

def generate_reading_test_json():
    """
    Generates the JSON for an IELTS Academic Reading Test with three passages.
    Each passage is unique in topic and has a different question type.
    """
    try:
        # Generate responses for each passage
        response1 = chain1.run({})
        response2 = chain2.run({})
        response3 = chain3.run({})

        # Parse each response into passage, questions, and answers
        passage1, questions1 = parse_response(response1)
        passage2, questions2 = parse_response(response2)
        passage3, questions3 = parse_response(response3)

        # Form the JSON output with separated passage, questions, and answers
        ielts_test_json = {
            "test": {
                "passages": [
                    {
                        "passage": "Passage 1",
                        "content": passage1,
                        "questions": questions1
                    },
                    {
                        "passage": "Passage 2",
                        "content": passage2,
                        "questions": questions2
                    },
                    {
                        "passage": "Passage 3",
                        "content": passage3,
                        "questions": questions3
                    }
                ]
            }
        }
        return ielts_test_json

    except Exception as e:
        print(f"Error generating reading test JSON: {e}")
        return None

# Streamlit interface for generating and displaying the reading test
st.title("IELTS Academic Reading Task Generator")

if st.button("Generate IELTS Reading Test"):
    with st.spinner("Generating IELTS Reading Test... This may take a few minutes."):

        # Generate JSON for the reading test
        ielts_test_json = generate_reading_test_json()

        # Check if JSON generation was successful
        if ielts_test_json:
            # Display generated topics to check uniqueness (text version)
            passages = ielts_test_json["test"]["passages"]
            st.subheader("Generated Topics for Each Passage (Text)")
            for i, passage in enumerate(passages):
                # Display the first line if content exists, otherwise show "(No content generated)"
                if passage['content']:
                    topic_line = passage['content'].splitlines()[0]
                else:
                    topic_line = "(No content generated)"
                st.markdown(f"**Passage {i + 1} Topic:** {topic_line}")

            # Display the full text output
            st.subheader("Generated IELTS Academic Reading Test (Text Format)")
            full_test = (f"IELTS Academic Reading Test\n\n"
                         f"READING PASSAGE 1:\n{passages[0]['content']}\n\n"
                         f"READING PASSAGE 2:\n{passages[1]['content']}\n\n"
                         f"READING PASSAGE 3:\n{passages[2]['content']}")
            st.markdown(full_test)

            # Option to download text output
            st.download_button(
                label="Download IELTS Reading Test (Text)",
                data=full_test,
                file_name="IELTS_Reading_Test.txt",
                mime="text/plain"
            )

            # Display JSON structure
            st.subheader("Generated IELTS Academic Reading Test (JSON Format)")
            st.json(ielts_test_json)

            # Option to download JSON output
            st.download_button(
                label="Download IELTS Reading Test (JSON)",
                data=json.dumps(ielts_test_json, indent=2),
                file_name="IELTS_Reading_Test.json",
                mime="application/json"
            )
        else:
            st.error("Error generating the IELTS Reading Test JSON.")
