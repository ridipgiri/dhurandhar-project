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

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if st.button("Process File"):
    if uploaded_file and api_key:
        rag = DhurandharRAG(api_key)
        with st.spinner("Extracting and Indexing..."):
            # Use the new extraction method
            text = rag.extract_text(uploaded_file)
            rag.chunk_text(text, chunk_size=chunk_size)
            rag.build_index()
            st.session_state.rag = rag
        st.success(f"File processed! Created {len(rag.chunks)} chunks.")
    else:
        st.warning("Please upload a file and ensure the API key is set.")

st.divider()

# Step 2: Chat
if st.session_state.rag:
    query = st.text_input("Ask a question about the uploaded file:")
    if query:
        with st.spinner("Analyzing context..."):
            answer = st.session_state.rag.query(query, k=k_val)
            st.markdown(f"### Answer:\n{answer}")
