import streamlit as st  # Import Streamlit
from writing1 import display_writing1_content  # Import the function to display content from writing1.py
from writing2 import display_writing2_content  # Import the function to display content from writing2.py
from vocabulary import display_vocabulary  # Import the vocabulary section
from grammar import display_grammar  # Import the grammar section

# Set the page configurations - This must be the first Streamlit command
st.set_page_config(page_title="IELTS Practice", layout="wide")

# Sidebar with a heading "Features"
st.sidebar.title("Features")

# Main options in the sidebar
option = st.sidebar.radio("Select an option:", ["Practice Session", "Mock Tests"])

# Initialize writing_option as None
writing_option = None

# Subsections under "Practice Session"
if option == "Practice Session":
    st.sidebar.subheader("Practice Sections")
    practice_option = st.sidebar.selectbox("Choose a section:", ["Reading", "Listening", "Writing", "Speaking", "Vocabulary", "Grammar"])

    # Subsections under "Writing"
    if practice_option == "Writing":
        st.sidebar.subheader("Writing tasks")
        writing_option = st.sidebar.selectbox("Choose a section:", ["Writing 1", "Writing 2"])

    # Subsections under "Reading"
    if practice_option == "Reading":
        st.sidebar.subheader("Reading tasks")
        reading_option = st.sidebar.selectbox("Choose a section:", ["Reading 1", "Reading 2", "Reading 3"])

    # Clear the session state for writing tasks when switching between "Writing 1" and "Writing 2"
    if 'previous_writing_option' not in st.session_state or st.session_state.previous_writing_option != writing_option:
        st.session_state.clear()  # Clear session state to reset variables
        st.session_state.previous_writing_option = writing_option

# Display content based on the selection
if option == "Practice Session":
    st.title(f"Practice Session: {practice_option}")
    st.write(f"Content for {practice_option} section will go here.")

    # Display content for "Writing" and "Writing 1" or "Writing 2"
    if practice_option == "Writing":
        if writing_option == "Writing 1":
            display_writing1_content()  # Call the function to display content from writing1.py
        elif writing_option == "Writing 2":
            display_writing2_content()  # Call the function to display content from writing2.py

    # Display content for Vocabulary
    if practice_option == "Vocabulary":
        display_vocabulary()  # Call the function to display vocabulary content

    # Display content for Grammar
    if practice_option == "Grammar":
        display_grammar()  # Call the function to display grammar content

elif option == "Mock Tests":
    st.title("Mock Tests")
    st.write("Mock test content will go here.")

# Simple styling for a clean interface
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .reportview-container .markdown-text-container {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
