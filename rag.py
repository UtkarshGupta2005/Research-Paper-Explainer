import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

doc_path=r"C:\AIProjects\ResearchPaperExplainer\NIPS-2017-attention-is-all-you-need-Paper.pdf"
if os.path.exists(doc_path):
    loader=PyMuPDFLoader(doc_path)
    documents=loader.load()
    print("PDF uploaded")
else:
    print("Wrong path")

splitter=SemanticChunker(OllamaEmbeddings(model="nomic-embed-text"),breakpoint_threshold_type="percentile")
chunks=splitter.split_documents(documents)

vector_db=Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name='research_papers',
    persist_directory='./Chroma_db'
)

#vector_db=Chroma(
#    persist_directory='./Chroma_db',
#    embedding_function=embedding_model
#)

QUERY_PROMPT=PromptTemplate(
    input_variable=["question"],
    template="""You are an AI language model assistant. 
    Your task is to generate a brief summary of the pdf which is any research paper. 
    You must summarize the most important and significant part of paper. 
    Original question: {question}"""
)

retriever=vector_db.as_retriever(search_type="mmr",
                                search_kwargs={
                                    "k":6,
                                    "fetch_k":20,
                                    "lambda_mult":0.7
                                })

def format_docs(docs):
    formatted=[]

    for doc in docs:
        source=doc.metadata.get("source","Unknown")
        page=doc.metadata.get("page","N/A")

        formatted.append(f"[Source:{source},Page:{page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


template="""You are an expert AI research assistant.

Use ONLY the provided context.

If the answer is not present in context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Provide:
1. Main objective
2. Core methodology
3. Architecture/design
4. Key innovations
5. Training details
6. Results
7. Limitations
8. Final conclusion

Use concise technical language."""
llm=ChatOllama(model="qwen2.5:3b",
               streaming=False)
prompt=ChatPromptTemplate.from_template(template)

chain=(
    {
     "context":retriever|format_docs,
     "question":RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

result=chain.invoke("Explain the uploaded pdf.")
print(result)

#compressor=FlashrankRerank()
#compressor.model_rebuild()
#compression_retriever=ContextualCompressionRetriever(
#    base_compressor=compressor,
#    base_retriever=retriever
#)