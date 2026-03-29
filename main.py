import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import io

class DhurandharRAG:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []

    def extract_text(self, uploaded_file):
        """Extracts text from PDF or TXT files.

        `uploaded_file` is expected to be a file-like object (for example,
        Streamlit's UploadedFile). This handles PDF via pypdf and falls back
        to reading text for other file types.
        """
        # Try to detect PDF by content-type or filename
        content_type = getattr(uploaded_file, "type", None)
        name = getattr(uploaded_file, "name", "")
        is_pdf = (content_type == "application/pdf") or name.lower().endswith('.pdf')

        # Ensure pointer at start
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if is_pdf:
            reader = PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)

        # Fallback: read as text
        data = uploaded_file.read()
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        return str(data)

    def chunk_text(self, text, chunk_size=500, chunk_overlap=50):
        text = text.strip()
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                for sep in separators:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap if end < len(text) else end
        self.chunks = chunks
        return chunks

    def build_index(self):
        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def query(self, question, k=2):
        query_vector = self.model.encode([question]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        context = "\n".join([self.chunks[i] for i in indices[0]])
        
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer only using the context above."
        
        completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content