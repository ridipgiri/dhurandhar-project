import streamlit as st
from main import DhurandharRAG
from groq import Groq

st.set_page_config(page_title="Dhurandhar RAG", layout="wide")
st.title("🧠 Dhurandhar Local RAG System")

# Safe rerun helper: `st.experimental_rerun` may be unavailable in some
# Streamlit versions. This function tries it and falls back to updating
# the query params which triggers a rerun.
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            import time
            params = st.experimental_get_query_params()
            params["_rerun"] = [str(time.time())]
            st.experimental_set_query_params(**params)
        except Exception:
            # Last-resort: stop the script so UI doesn't continue in bad state
            st.stop()

# Sidebar for API Key
with st.sidebar:
    # Provide your Groq API Key here or set the environment variable `GROQ_API_KEY`
    api_key = st.text_input("Groq API Key", value="", placeholder="Enter your Groq API key", type="password")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    k_val = st.slider("Context Chunks (k)", 1, 5, 2)
    if st.button("Validate API Key"):
        if not api_key:
            st.error("Please enter an API key to validate.")
        elif not api_key.startswith('gsk_'):
            st.warning("The key doesn't look like a Groq key (missing 'gsk_' prefix). Will still attempt to use it.")
        else:
            try:
                Groq(api_key=api_key)
                st.success("API key appears to instantiate correctly (no network validation).")
            except Exception as e:
                st.error(f"Failed to instantiate Groq client: {e}")

if "rag" not in st.session_state:
    st.session_state.rag = None

# Text input and file upload
st.subheader("Add Content")
raw_text = st.text_area("Paste document content here (optional)", height=150)
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Add to Knowledge Base"):
        if not api_key:
            st.error("Please provide an API key in the sidebar.")
        else:
            # If a RAG exists but the API key changed, update the client on the existing RAG.
            if st.session_state.rag:
                try:
                    current_key = getattr(st.session_state.rag.client, 'api_key', None)
                except Exception:
                    current_key = None
                if api_key and api_key != current_key:
                    try:
                        st.session_state.rag.client = Groq(api_key=api_key)
                        st.info("Updated API key for existing knowledge base.")
                    except Exception as e:
                        st.error(f"Could not update API key on existing RAG: {e}")
            rag = st.session_state.rag or DhurandharRAG(api_key)
            total_new = 0
            # process raw text
            if raw_text and raw_text.strip():
                new_chunks = rag.chunk_text(raw_text, chunk_size=chunk_size, append=True)
                rag.add_chunks(new_chunks)
                total_new += len(new_chunks)

            # process uploaded files
            if uploaded_files:
                for f in uploaded_files:
                    text = rag.extract_text(f)
                    if text and text.strip():
                        new_chunks = rag.chunk_text(text, chunk_size=chunk_size, append=True)
                        rag.add_chunks(new_chunks)
                        total_new += len(new_chunks)

            st.session_state.rag = rag
            st.success(f"Added {total_new} new chunks. Total chunks: {len(rag.chunks)}")

with col2:
    if st.button("Reset Knowledge Base"):
        st.session_state.rag = None
        st.success("Knowledge base reset.")

st.divider()

# Chat section with conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def render_messages():
    for msg in st.session_state.messages:
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Bot:** {content}")

if st.session_state.rag:
    st.subheader("Chat with your documents")
    render_messages()
    # Use a form so the input can be cleared automatically on submit
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your question:", key='chat_input')
        submit = st.form_submit_button("Send")
    if submit and user_input and user_input.strip():
        # append user message
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        rag = st.session_state.rag
        with st.spinner("Thinking..."):
            try:
                answer = rag.query(user_input, k=k_val)
            except Exception as e:
                answer = f"Error during query: {e}"
        st.session_state.messages.append({'role': 'bot', 'content': answer})
        safe_rerun()
else:
    st.info("No knowledge base yet. Add text or files above.")
