"""
RAG Agent Backend — FastAPI
Run with:
  pip install fastapi uvicorn python-multipart
  uvicorn server:app --reload --port 8000
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR  = "./Chroma_db"
COLLECTION  = "research_papers"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "qwen2.5:3b"

TEMPLATE = """You are an expert AI research assistant.
Use ONLY the provided context. If information is absent say "Not found in document."

Context:
{context}

Question: {question}

Answer clearly and technically. Quote numbers, names, and results from the context where available."""

# ── App state ─────────────────────────────────────────────────────────────────

class State:
    chain = None
    pdf_name: str = None
    chunk_count: int = 0

state = State()

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_chain(llm_model: str, embed_model: str):
    embedding = OllamaEmbeddings(model=embed_model)
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
        collection_name=COLLECTION,
    )
    base_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12, "lambda_mult": 0.7},
    )
    mq_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Generate 5 search queries to retrieve all important sections of a research paper needed to answer:

{question}

Output ONLY 5 queries, one per line, no numbering:""",
    )
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=ChatOllama(model=llm_model, streaming=False),
        prompt=mq_prompt,
    )

    def format_docs(docs):
        seen, parts = set(), []
        for doc in docs:
            key = doc.page_content[:120]
            if key in seen:
                continue
            seen.add(key)
            page = doc.metadata.get("page", "N/A")
            parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n".join(parts)

    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = ChatOllama(model=llm_model, streaming=True)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {
        "loaded": state.chain is not None,
        "pdf_name": state.pdf_name,
        "chunk_count": state.chunk_count,
    }


@app.post("/upload")
async def upload(
    # FIX 1: Use Form(...) so FastAPI reads these from multipart form data
    file: UploadFile = File(...),
    llm_model: str   = Form(LLM_MODEL),
    embed_model: str = Form(EMBED_MODEL),
    chunk_size: int  = Form(800),
    chunk_overlap: int = Form(100),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # FIX 2: Read the file content first, THEN delete the temp file
    # On Windows, open file handles block deletion — close before unlinking
    content = await file.read()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        # File is now closed — safe to read with PyMuPDF on Windows
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
    except Exception as e:
        raise HTTPException(500, f"Failed to read PDF: {e}")
    finally:
        # FIX 3: Always clean up temp file, even on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not documents:
        raise HTTPException(400, "PDF appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Rebuild vector DB
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    embedding = OllamaEmbeddings(model=embed_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    state.chain     = build_chain(llm_model, embed_model)
    state.pdf_name  = file.filename
    state.chunk_count = len(chunks)

    return {
        "status": "ok",
        "pdf_name": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
    }


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    if state.chain is None:
        raise HTTPException(400, "No document loaded. Upload a PDF first.")

    def token_stream():
        for token in state.chain.stream(req.question):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.delete("/reset")
def reset():
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    state.chain     = None
    state.pdf_name  = None
    state.chunk_count = 0
    return {"status": "reset"}