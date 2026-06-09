import os
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever

doc_path=r"C:\AIProjects\ResearchPaperExplainer\NIPS-2017-attention-is-all-you-need-Paper.pdf"
Chroma_dir="./Chroma_db"
Collection="research_papers"
llm_model="qwen2.5:7b"
RERANK=True

embedding_model=OllamaEmbeddings(model="nomic-embed-text")
db_exists=Path(Chroma_dir).exists() and any(Path(Chroma_dir).iterdir())

if db_exists:
    print("Loading existing Chroma DB from disk...")
    vector_db = Chroma(
        persist_directory=Chroma_dir,
        embedding_function=embedding_model,
        collection_name=Collection,
    )
else:
    print("Building vector DB — this runs once, subsequent runs are instant.")
 
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"PDF not found: {doc_path}")
 
    loader = PyMuPDFLoader(doc_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100,separators=["\n\n","\n","."," ",""])
    chunks=splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=Collection,
        persist_directory=Chroma_dir,
    )
    print("Vector DB built and saved.")

llm=ChatOllama(model=llm_model, streaming=True)

base_retriever=vector_db.as_retriever(search_type="mmr",
                                 search_kwargs={"k":5,"fetch_k":12,"lambda_mult":0.7})

multi_query_prompt=PromptTemplate(input_variable=["question"],
                                  template="""You are an AI assistant helping retrieve information from a research paper.
Generate 5 different search queries that together would retrieve all the important 
sections of the paper needed to answer this question.
 
Focus on: objective, methodology, architecture, innovations, training, results, limitations.
 
Original question: {question}
 
Output ONLY the 5 queries, one per line, no numbering or bullets:""")

retriever=MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOllama(model=llm_model,streaming=False),
    prompt=multi_query_prompt
)

def format_docs(docs):
    seen=set()
    parts=[]
    for doc in docs:
        key=doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        page=doc.metadata.get("page","N/A")
        parts.append(f"[Page: {page}]\n{doc.page_content}")
        return "\n\n".join(parts)
    
TEMPLATE = """You are an expert AI research assistant.
Use ONLY the provided context. If a section's information is genuinely absent, 
say "Not covered in the retrieved sections" for that point only.
 
Context:
{context}
 
Question: {question}
 
Structure your answer as:
1. Main Objective
2. Core Methodology
3. Architecture / Design
4. Key Innovations
5. Training Details
6. Results & Benchmarks
7. Limitations
8. Conclusion
 
Be specific and technical. Quote numbers and names from the context where available."""

prompt=ChatPromptTemplate.from_template(TEMPLATE)

llm=ChatOllama(model=llm_model, streaming=False)
chain=(
    {"context":retriever|format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm 
    | StrOutputParser()
)

print("\n" + "─" * 60)
question = "Give a complete summary of this research paper."
 
for token in chain.stream(question):
    print(token, end="", flush=True)
 
print("\n" + "─" * 60)