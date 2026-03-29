import streamlit as st
from main import DhurandharRAG

st.set_page_config(page_title="Dhurandhar RAG", layout="wide")
st.title("🧠 Dhurandhar Local RAG System")

# Sidebar for API Key
with st.sidebar:
    # Provide your Groq API Key here or set the environment variable `GROQ_API_KEY`
    api_key = st.text_input("Groq API Key", value="", placeholder="Enter your Groq API key", type="password")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    k_val = st.slider("Context Chunks (k)", 1, 5, 2)

if "rag" not in st.session_state:
    st.session_state.rag = None

# Step 1: Input Text
raw_text = st.text_area("Paste Document Content", height=200)

if st.button("Initialize Knowledge Base"):
    if raw_text and api_key:
        rag = DhurandharRAG(api_key)
        with st.spinner("Processing..."):
            rag.chunk_text(raw_text, chunk_size=chunk_size)
            rag.build_index()
            st.session_state.rag = rag
        st.success(f"Indexed {len(rag.chunks)} chunks!")
    else:
        st.error("Please provide text and API key.")

st.divider()

# Step 2: Chat
if st.session_state.rag:
    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.query(query, k=k_val)
            st.markdown(f"**Answer:**\n{answer}")
