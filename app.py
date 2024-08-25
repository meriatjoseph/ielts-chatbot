import streamlit as st  #import statement for streamlit 

# Set the page configurationsss
st.set_page_config(page_title="IELTS Practice", layout="wide")

# Sidebar with a heading "Features"
st.sidebar.title("Features")

# Main options in the sidebar
option = st.sidebar.radio("Select an option:", ["Practice Session", "Mock Tests"])

# Subsections under "Practice Session"
if option == "Practice Session":
    st.sidebar.subheader("Practice Sections")
    practice_option = st.sidebar.selectbox("Choose a section:", ["Reading", "Listening", "Writing", "Speaking"])

# Display content based on the selection
if option == "Practice Session":
    st.title(f"Practice Session: {practice_option}")
    st.write(f"Content for {practice_option} section will go here.")
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
