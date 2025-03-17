import sqlite3
import uuid
from conversation.data_base import create_chat_session, save_message, get_chat_history, create_user_profile, save_chat_title
from conversation.llm_api import generate_llm_response, generate_chat_title_from_history

def start_chat(user_id, role, name="User"):
    session_id = str(uuid.uuid4())
    create_chat_session(session_id, user_id, role)
    create_user_profile(user_id, name)  # Save user profile

    return {"session_id": session_id, "message": f"Chat started as {role}, {name}!"}


def process_chat(session_id, user_message):
    # Generate LLM response with full context
    bot_response = generate_llm_response(session_id, user_message)

    # Save both user and bot messages to DB
    save_message(session_id, "User", user_message)
    save_message(session_id, "Bot", bot_response)
    
    return {"response": bot_response}


def fetch_chat_history(session_id):
    return get_chat_history(session_id)



def process_chat(session_id, user_message):
    # Generate LLM response with full context
    bot_response = generate_llm_response(session_id, user_message)

    # Save both user and bot messages to DB
    save_message(session_id, "User", user_message)
    save_message(session_id, "Bot", bot_response)

    # Auto-generate title if missing
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    c.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,))
    result = c.fetchone()
    conn.close()

    if result and (not result[0] or result[0].strip() == ""):
        title = generate_chat_title_from_history(session_id)
        save_chat_title(session_id, title)

    return {"response": bot_response}