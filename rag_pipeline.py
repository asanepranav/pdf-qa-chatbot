"""
rag_pipeline.py — Core RAG logic

Flow:
  PDF → chunks → HuggingFace embeddings → FAISS index
  Question → embed → retrieve top-k chunks → Gemini → answer + sources
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer based only on the context above.
- If the answer isn't in the context, say "I couldn't find this in the document."
- Be concise but complete.
- Cite relevant parts naturally in your answer.

Answer:"""


def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_answer(vectorstore, question: str) -> dict:
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    source_docs = retriever.invoke(question)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    sources = []
    for doc in source_docs:
        sources.append({
            "text": doc.page_content,
            "page": (doc.metadata.get("page") or 0) + 1
        })

    return {
        "answer": answer,
        "sources": sources
    }
