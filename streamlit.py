# NOTE: Avoid running this file with `python streamlit.py` because that
# would shadow the installed `streamlit` package. Run with the Streamlit
# CLI instead: `streamlit run streamlit.py` or use `app.py`.
import streamlit as st
from main import DhurandharRAG

st.set_page_config(page_title="Dhurandhar RAG", layout="wide")
st.title("📂 Dhurandhar File RAG")

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

# Chat
if st.session_state.rag:
    st.subheader("Ask a question")
    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Analyzing context..."):
            answer = st.session_state.rag.query(query, k=k_val)
            st.markdown(f"### Answer:\n{answer}")
