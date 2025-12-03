import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import PyPDF2

# Page Configuration
st.set_page_config(
    page_title="Noah Haag - AI Resume Agent",
    page_icon="📄",
    layout="centered"
)

# Load Environment Variables
load_dotenv()

# Constants
MODEL_ID = "gemini-2.0-flash-001"
RESUME_PATH = "public/Resume.pdf"

@st.cache_resource
def load_resume_content():
    """Loads and extracts text from the resume PDF. Cached for performance."""
    if not os.path.exists(RESUME_PATH):
        return None
    
    text = ""
    try:
        with open(RESUME_PATH, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text

@st.cache_resource
def get_gemini_client():
    """Initializes the Gemini client. Cached so it persists."""
    api_key = os.getenv("GOOGLE_API_KEY")
    # Check Streamlit secrets if env var not found (for cloud deployment)
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

# --- Main UI ---

st.title("Talk to Noah's Resume 🤖")
st.markdown("Ask questions about my experience, skills, or background.")

# Initialize Client
# We use cache_resource above, so this doesn't re-instantiate the client object repeatedly
client = get_gemini_client()

if not client:
    st.warning("Please set your GOOGLE_API_KEY in the environment or Streamlit secrets.")
    st.stop()

# Load Resume
resume_text = load_resume_content()
if not resume_text:
    st.error("Could not load resume content. Please ensure public/Resume.pdf exists.")
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    # System Prompt Injection (Hidden from UI)
    system_instruction = f"""You are a helpful assistant representing Noah Haag.
    You have access to Noah's resume context below.
    Answer questions about Noah's experience, skills, and background based strictly on this information.
    Be concise, professional, and engaging.
    
    RESUME CONTEXT:
    {resume_text}
    """
    st.session_state.chat_session = client.chats.create(
        model=MODEL_ID,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )
    )

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    try:
        # We use the session_state chat object which persists across reruns
        response = st.session_state.chat_session.send_message(prompt)
        
        # Add assistant message to UI
        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        # If the session somehow died (e.g. timeout), we can try to recover or just show error
        st.error(f"An error occurred: {e}")
        # Optional: Force a reload of the page if critical
        # st.rerun()
