import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import PyPDF2
import json

# Page Configuration
st.set_page_config(
    page_title="Noah Haag | Interactive Resume",
    page_icon="📄", # You can change this to a custom emoji or image
    layout="centered"
)

# Load Environment Variables
load_dotenv()

# Constants
MODEL_ID = "gemini-2.0-flash-001"
BACKUP_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]
RESUME_PATH = "public/Resume.pdf"
PROFILE_PHOTO_PATH = "public/profile.png" # Assuming a PNG, change if JPG
BRAIN_PATH = "data/brain.json"

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
def load_brain_content():
    """Loads the 'brain' JSON file containing hidden context."""
    if not os.path.exists(BRAIN_PATH):
        return None
    
    try:
        with open(BRAIN_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading Brain JSON: {e}")
        return None

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

# Sidebar Content
with st.sidebar:
    if os.path.exists(PROFILE_PHOTO_PATH):
        st.image(PROFILE_PHOTO_PATH, use_column_width=True)
    st.title("Noah Haag")
    st.markdown("---")
    st.subheader("Connect with me:")
    st.markdown("📧 [Your Email](mailto:noahhaag1998@gmail.com)") # UPDATE THIS
    st.markdown("👔 [LinkedIn Profile](https://www.linkedin.com/in/noah-haag-961691161/)") # UPDATE THIS
    st.markdown("🐙 [GitHub Profile](https://github.com/NoahHaag)") # UPDATE THIS
    st.markdown("---")
    
    if os.path.exists(RESUME_PATH):
        with open(RESUME_PATH, "rb") as file:
            btn = st.download_button(
                label="Download Full Resume",
                data=file,
                file_name="Noah_Haag_Resume.pdf",
                mime="application/pdf"
            )

st.title("Talk to My Resume 🤖")
st.markdown("Ask questions about my experience, skills, or background.")

# Initialize Client
client = get_gemini_client()

if not client:
    st.warning("Please set your GOOGLE_API_KEY in the environment or Streamlit secrets.")
    st.stop()

# Load Resume
resume_text = load_resume_content()
if not resume_text:
    st.error("Could not load resume content. Please ensure public/Resume.pdf exists.")
    st.stop()

# Load Brain
brain_content = load_brain_content()
if not brain_content:
    # Optional: Warn if brain is missing, or just proceed without it
    # st.warning("Brain content not found. Some answers may be limited.")
    brain_text = "{}" # Empty JSON if missing
else:
    brain_text = json.dumps(brain_content, indent=2)

# System Prompt Injection (Hidden from UI)
system_instruction = f"""You are Noah Haag. Your goal is to represent yourself based on the provided context. 

You have access to two sources of information:
1. **RESUME CONTEXT**: Your official professional history. Prioritize this for factual questions about dates, roles, and hard skills.
2. **HIDDEN CONTEXT (BRAIN)**: Your deeper thoughts, personality, logistics (availability, relocation), and behavioral stories (STAR method). Use this to answer questions about "soft skills", "failures", "leadership", or "why you love tech".

**INSTRUCTIONS:**
- Answer questions based STRICTLY on the context provided below.
- Keep your answers concise and professional, typically under 3-4 sentences, unless the user asks for more detail.
- If the answer is NOT in your Resume or Brain, politely state that you do not have that specific information. DO NOT invent information.
- If the user asks for your contact information or how to hire you, provide your email, LinkedIn, and GitHub links (from your sidebar information, if available) clearly and enthusiastically.
- Use the 'marine_biology_context' to explain how your scientific background enhances your engineering skills (e.g., rigor, adaptability) when asked about your career transition or background.
- When asked for a fun fact, pick one randomly from the 'fun_facts' list in the Brain.
- Be engaging and personable, using the "Voice" found in the 'technical_opinions' or 'hobbies' section of the Brain if appropriate.

**RESUME CONTEXT:**
{resume_text}

**HIDDEN CONTEXT (BRAIN):**
{brain_text}
"""

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
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

# Quick Action Buttons
col1, col2, col3 = st.columns(3)
if col1.button("Tell me about your skills"):
    st.session_state.messages.append({"role": "user", "content": "Tell me about your skills"})
    st.session_state.last_button_question = "Tell me about your skills"
if col2.button("What's your education history?"):
    st.session_state.messages.append({"role": "user", "content": "What's your education history?"})
    st.session_state.last_button_question = "What's your education history?"
if col3.button("How can I contact you?"):
    st.session_state.messages.append({"role": "user", "content": "How can I contact you?"})
    st.session_state.last_button_question = "How can I contact you?"

# Chat Input
prompt = st.chat_input("Ask me anything...")

# Handle button clicks
if "last_button_question" in st.session_state and st.session_state.last_button_question and not prompt:
    prompt = st.session_state.last_button_question
    st.session_state.last_button_question = None # Clear after use

if prompt:
    # Add user message to UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.spinner('Noah is thinking...'): # Added spinner for visual feedback
        try:
            # We use the session_state chat object which persists across reruns
            response = st.session_state.chat_session.send_message(prompt)
        except Exception as e:
            # BROAD ERROR HANDLING: Catch 429 (Rate Limit), 404 (Not Found), 500, etc.
            # If ANY error happens, we try backups, then fallback to offline.
            error_msg = str(e)
            print(f"Primary model failed: {error_msg}") # Log to console for debugging

            success_fallback = False
            for model in BACKUP_MODELS:
                try:
                    # Re-create session with new model and existing history
                    formatted_history = []
                    for msg in st.session_state.messages:
                        role = "model" if msg["role"] == "assistant" else "user"
                        formatted_history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
                    
                    st.warning(f"Primary model unavailable. Switching to backup: {model}...")
                    
                    # Create a TEMP session first. Only update state if it works.
                    temp_session = client.chats.create(
                        model=model,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.7,
                        ),
                        history=formatted_history
                    )
                    # Try sending message
                    response = temp_session.send_message(prompt)
                    
                    # If we get here, it worked! Update session state.
                    st.session_state.chat_session = temp_session
                    success_fallback = True
                    break # Stop loop if successful
                except Exception as fallback_e:
                    print(f"Backup model {model} failed: {fallback_e}")
                    continue # Try next backup

            if not success_fallback:
                # --- EMERGENCY OFFLINE MODE ---
                st.toast("⚠️ API Unavailable. Switching to Offline Mode.", icon="📡")
                
                def get_fallback_response(query, resume_txt, brain_data):
                    query = query.lower()
                    
                    # 1. Direct Intent Matching (Fast & Free)
                    if "contact" in query or "email" in query or "reach" in query:
                        return "You can reach me via:\n- **Email**: noahhaag1998@gmail.com\n- **LinkedIn**: [Profile](https://www.linkedin.com/in/noah-haag-961691161/)\n- **GitHub**: [NoahHaag](https://github.com/NoahHaag)"
                    
                    if "skill" in query or "stack" in query:
                            # Try to extract skills from Brain or Resume
                            if isinstance(brain_data, dict) and "skills" in brain_data:
                                return f"**My Key Skills:**\n{json.dumps(brain_data['skills'], indent=2)}"
                            return "I have experience with Python, AI/ML, Streamlit, and Data Analysis. (Check my resume for the full list!)"
                            
                    if "education" in query or "university" in query or "degree" in query:
                        # Simple extraction heuristic
                        return "I have a background in Marine Biology and Computer Science. Please download my resume for the full education history."

                    # 2. Keyword Search (Vector-lite)
                    # Split resume into paragraphs
                    paragraphs = [p.strip() for p in resume_txt.split('\n') if len(p.strip()) > 20]
                    query_words = set(query.split())
                    
                    best_match = None
                    best_score = 0
                    
                    for p in paragraphs:
                        score = sum(1 for w in query_words if w in p.lower())
                        if score > best_score:
                            best_score = score
                            best_match = p
                    
                    if best_match and best_score > 0:
                        return f"*(Offline Mode)* Here is something relevant from my resume:\n\n> {best_match}"
                    
                    return "I'm currently experiencing high traffic and couldn't process that specific question. Please try asking about my **Skills**, **Education**, or **Contact Info**, or download my resume from the sidebar!"

                # Generate offline response
                offline_response_text = get_fallback_response(prompt, resume_text, brain_content)
                
                # Add to chat history so it looks normal
                st.session_state.messages.append({"role": "assistant", "content": offline_response_text})
                
                # Display
                with st.chat_message("assistant"):
                    st.markdown(offline_response_text)
                    
                # Stop further execution to avoid error display
                st.stop()
            
        # Add assistant message to UI (if response was successful)
        if 'response' in locals() and response:
            with st.chat_message("assistant"):
                st.markdown(response.text)
                
                # --- Artifact Rendering Logic ---
                if brain_content and "artifacts" in brain_content:
                    for artifact_name, artifact_data in brain_content["artifacts"].items():
                        # Check if any trigger word appears in the response OR the user's prompt
                        if any(keyword.lower() in response.text.lower() for keyword in artifact_data["trigger_words"]) or \
                           any(keyword.lower() in prompt.lower() for keyword in artifact_data["trigger_words"]):
                            
                            # Only show if image exists to avoid broken UI
                            if os.path.exists(artifact_data["image_path"]):
                                with st.expander(f"👀 View Artifact: {artifact_data['caption']}"):
                                    st.image(artifact_data["image_path"], caption=artifact_data["caption"])
                                    if "link" in artifact_data and artifact_data["link"]:
                                        st.markdown(f"🔗 [View Project on GitHub]({artifact_data['link']})")
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
