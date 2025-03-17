import sqlite3
import requests
import os
from dotenv import load_dotenv
from conversation.data_base import get_chat_history, get_user_profile, save_chat_title

# Load API key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = GROQ_KEY
DB_NAME = "chat.db"

# Generate LLM response
def generate_llm_response(session_id, user_message):
    from conversation.data_base import get_user_profile
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Get user_id and role from session
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT user_id, role FROM chat_sessions WHERE session_id = ?", (session_id,))
    result = c.fetchone()
    conn.close()

    user_id, role = result if result else (None, "default")
    profile = get_user_profile(user_id) if user_id else None
    user_name = profile["name"] if profile else "User"

    # Define system prompt based on role
    role_prompts = {
        "interviewer": f"You are a professional job interviewer conducting a mock interview with {user_name}. Ask relevant interview questions and provide feedback.",
        "doctor": f"You are a knowledgeable and friendly doctor consulting with {user_name}. Provide medical advice in a clear and compassionate way.",
        "friend": f"You are a casual, friendly conversational partner chatting with {user_name}. Be relaxed and supportive.",
        "coach": f"You are a supportive coach helping {user_name} improve their verbal clarity. Give feedback on tone, clarity, and suggestions for improvement.",
        "default": f"You are a helpful chatbot chatting with {user_name}."
    }
    system_prompt = role_prompts.get(role.lower(), role_prompts["default"])

    # Prepare messages for LLM
    history = get_chat_history(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = "user" if msg["sender"] == "User" else "assistant"
        messages.append({"role": role, "content": msg["message"]})
    messages.append({"role": "user", "content": user_message})

    data = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json()}"

def generate_conversation_feedback(session_id):
    # Get full chat history (both bot and user)
    history = get_chat_history(session_id)
    combined_text = "\n".join([f"{msg['sender'].capitalize()}: {msg['message']}" for msg in history])

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a conversation specialist analyzing how a person engaged in a conversation. Below is the full dialogue.

Conversation:
{combined_text}

Analyze the user's performance based on their responses in the conversation. Focus your feedback on:
- Clarity and coherence of the user's replies
- Tone, engagement, and relevance in their responses
- Grammar and fluency, if notable
- Appropriateness and effectiveness of their communication

Provide a detailed evaluation of how the user communicated and offer specific suggestions for improvement.

Do not mention that the user was speaking to a bot.
    """.strip()

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a conversation expert evaluating how a user engaged in a dialogue."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 700
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.json()}"

# Generate chat title
def generate_chat_title_from_history(session_id):
    history = get_chat_history(session_id)
    combined_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in history])

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Based on the following conversation, generate a concise and relevant title (max 8 words) that reflects the main topic discussed:

    {combined_text}
    """

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "Untitled Chat"
