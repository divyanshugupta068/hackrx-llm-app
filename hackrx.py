import os
import requests
import tempfile
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import pinecone
import openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import uvicorn

# ------------------------
# 📦 Load environment variables from .env
# ------------------------
load_dotenv()

# ------------------------
# 🔐 API Key Setup (from .env)
# ------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
INDEX_NAME = "hackrx-index"

openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------
# 🔍 Vector DB Setup
# ------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)
index = pinecone.Index(INDEX_NAME)

# ------------------------
# 🔧 Embedding Model
# ------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# ⚙ FastAPI App Init
# ------------------------
app = FastAPI()

# ------------------------
# 🧾 Input Schema
# ------------------------
class InputData(BaseModel):
    documents: str
    questions: List[str]

# ------------------------
# 📥 Download Document
# ------------------------
def download_file(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to download file.")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name

# ------------------------
# 📄 Extract Text from PDF
# ------------------------
def extract_text(filepath):
    import fitz  # PyMuPDF
    doc = fitz.open(filepath)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# ------------------------
# ✂ Chunking Function
# ------------------------
def chunk_text(text, chunk_size=300):
    import textwrap
    return textwrap.wrap(text, chunk_size)

# ------------------------
# 📚 Store Embeddings in Pinecone
# ------------------------
def store_embeddings(text_chunks):
    embeddings = model.encode(text_chunks).tolist()
    vectors = [(f"chunk-{i}", emb, {"text": chunk}) for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings))]
    index.upsert(vectors)

# ------------------------
# 🔎 Search Chunks using Pinecone
# ------------------------
def search_chunks(question, top_k=3):
    q_emb = model.encode([question])[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

# ------------------------
# 🧠 Generate Answer using GPT
# ------------------------
def generate_gpt_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
Answer the following insurance-related question based on the context below:

Context:
{context}

Question: {question}
Answer:"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------
# 🚀 Main API Endpoint
# ------------------------
@app.post("/api/v1/hackrx/run")
def query_doc(input_data: InputData, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    
    filepath = download_file(input_data.documents)
    text = extract_text(filepath)
    
    chunks = chunk_text(text)
    store_embeddings(chunks)

    final_answers = []
    for question in input_data.questions:
        relevant_chunks = search_chunks(question)
        answer = generate_gpt_answer(question, relevant_chunks)
        final_answers.append(answer)

    return {"answers": final_answers}

# ------------------------
# 🖥 Local/Render Startup
# ------------------------
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("hackrx:app", host="0.0.0.0", port=port)
