"""
core/vector_store.py
--------------------
Chroma-based vector store for transcript RAG with automatic device fallback.
"""

import logging
import os

import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHROMA_DIR: str = os.getenv("CHROMA_DIR", "vector_db")
COLLECTION_NAME: str = "meeting_transcript"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embeddings: HuggingFaceEmbeddings | None = None


def _detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedding() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        device = _detect_device()
        logger.info("Loading embedding model '%s' on '%s'", EMBEDDING_MODEL, device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )
    return _embeddings


def build_vector_store(
    transcript: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> Chroma:
    """
    Split the transcript and index it into an in-memory Chroma collection.

    A fresh collection is created for each analysis run (no stale data).
    """
    logger.info("Building vector store …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embedding()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )
    logger.info("Vector store ready (%d chunks).", len(docs))
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing persisted Chroma vector store from disk."""
    embeddings = get_embedding()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def get_retriever(vector_store: Chroma, k: int = 5):
    """Return a similarity-based retriever over the given vector store."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
