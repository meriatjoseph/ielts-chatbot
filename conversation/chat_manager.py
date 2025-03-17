import uuid
from conversation.data_base import (
    create_chat_session,
    save_message,
    get_chat_history,
    create_user_profile,
    save_chat_title
)
from conversation.llm_api import generate_llm_response, generate_chat_title_from_history
import sqlite3

# Start chat session
def start_chat(user_id, role, name="User"):
    session_id = str(uuid.uuid4())
    create_chat_session(session_id, user_id, role)
    create_user_profile(user_id, name)
    return {"session_id": session_id, "message": f"Chat started as {role}, {name}!"}

# Process chat interaction
def process_chat(session_id, user_message):
    bot_response = generate_llm_response(session_id, user_message)
    save_message(session_id, "User", user_message)
    save_message(session_id, "Bot", bot_response)

    # Generate title if not set
    conn = sqlite3.connect("chat.db")
    c = conn.cursor()
    c.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,))
    result = c.fetchone()
    conn.close()

    if result and (not result[0] or result[0].strip() == ""):
        title = generate_chat_title_from_history(session_id)
        save_chat_title(session_id, title)

    return {"response": bot_response}

# Fetch chat history
def fetch_chat_history(session_id):
    return get_chat_history(session_id)
