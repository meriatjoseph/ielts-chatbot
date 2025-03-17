import sqlite3

DB_NAME = "chat.db"

# Initialize the database and tables
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # User profiles table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT
        )
    ''')

    # Chat sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            role TEXT,
            title TEXT
        )
    ''')

    # Messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            session_id TEXT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    ''')

    conn.commit()
    conn.close()

# Create or update user profile
def create_user_profile(user_id, name, preferences=""):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO user_profiles (user_id, name, preferences) VALUES (?, ?, ?)",
        (user_id, name, preferences)
    )
    conn.commit()
    conn.close()

# Retrieve user profile
def get_user_profile(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, preferences FROM user_profiles WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return {"name": result[0], "preferences": result[1]}
    return None

# Create chat session
def create_chat_session(session_id, user_id, role):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chat_sessions (session_id, user_id, role, title) VALUES (?, ?, ?, '')", 
              (session_id, user_id, role))
    conn.commit()
    conn.close()

# Save a chat message
def save_message(session_id, sender, message):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, sender, message) VALUES (?, ?, ?)", 
              (session_id, sender, message))
    conn.commit()
    conn.close()

# Retrieve chat history
def get_chat_history(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT sender, message, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp", 
              (session_id,))
    result = c.fetchall()
    conn.close()
    return [{"sender": row[0], "message": row[1], "timestamp": row[2]} for row in result]

# Save generated chat title
def save_chat_title(session_id, title):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (title, session_id))
    conn.commit()
    conn.close()

# Retrieve all sessions for a user
def get_user_sessions(user_id):
    from conversation.llm_api import generate_chat_title_from_history, save_chat_title

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT session_id, title FROM chat_sessions WHERE user_id = ?", (user_id,))
    sessions = c.fetchall()
    conn.close()

    chat_list = []
    for session_id, title in sessions:
        if not title or title.strip() == "":
            title = generate_chat_title_from_history(session_id)
            save_chat_title(session_id, title)
        chat_list.append({"session_id": session_id, "title": title})
    
    return chat_list
