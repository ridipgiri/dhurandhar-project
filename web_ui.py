# Web UI entrypoint (renamed to avoid shadowing the `streamlit` package)
# Run with: `streamlit run app.py` (preferred) or `streamlit run web_ui.py`
import streamlit as st
from main import DhurandharRAG

st.set_page_config(page_title="Dhurandhar RAG", layout="wide")
st.title("📂 Dhurandhar File RAG")

# Safe rerun helper
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
            st.stop()

# Sidebar
with st.sidebar:
    # Provide your Groq API Key here or set the environment variable `GROQ_API_KEY`
    api_key = st.text_input("Groq API Key", value="", placeholder="Enter your Groq API key", type="password")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    k_val = st.slider("Context Chunks (k)", 1, 5, 2)
    st.info("Supported: .pdf, .txt")

if "rag" not in st.session_state:
    st.session_state.rag = None

# Add text and multiple files
st.subheader("Add Content")
raw_text = st.text_area("Paste document content here (optional)", height=150)
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Add to Knowledge Base"):
        if not api_key:
            st.error("Please provide an API key in the sidebar.")
        else:
            rag = st.session_state.rag or DhurandharRAG(api_key)
            total_new = 0
            if raw_text and raw_text.strip():
                new_chunks = rag.chunk_text(raw_text, chunk_size=chunk_size, append=True)
                rag.add_chunks(new_chunks)
                total_new += len(new_chunks)
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

# Chat with history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def render_msgs():
    for m in st.session_state.messages:
        if m['role'] == 'user':
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Bot:** {m['content']}")

if st.session_state.rag:
    st.subheader("Chat with your documents")
    render_msgs()
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your question:", key='chat_input')
        submit = st.form_submit_button("Send")
    if submit and user_input and user_input.strip():
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        rag = st.session_state.rag
        with st.spinner("Thinking..."):
            try:
                answer = rag.query(user_input, k=k_val)
            except Exception as e:
                answer = f"Error: {e}"
        st.session_state.messages.append({'role': 'bot', 'content': answer})
        safe_rerun()
else:
    st.info("No knowledge base yet. Add text or files above.")
