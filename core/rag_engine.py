"""
core/rag_engine.py
------------------
Transcript-grounded RAG chat chain using LangChain + Mistral AI.
"""

import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from core.vector_store import build_vector_store, get_retriever

logger = logging.getLogger(__name__)

_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

_SYSTEM_PROMPT = """\
You are an expert meeting assistant. Your job is to answer the user's question \
based ONLY on the meeting transcript context provided below.

Guidelines:
- If the answer is not in the context, say: \
  "I could not find this information in the meeting transcript."
- Be concise and precise. Avoid padding or filler text.
- If quoting a speaker, attribute the quote clearly.
- If the question is ambiguous, ask for clarification.

Context from the meeting transcript:
{context}
"""


def _get_llm(temperature: float = 0.3) -> ChatNVIDIA:
    """Return a configured NVIDIA NIM LLM instance."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. Add it to your .env file."
        )
    return ChatNVIDIA(
        model=_NVIDIA_MODEL,
        api_key=api_key,
        temperature=temperature,
    )


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(transcript: str, k: int = 5):
    """
    Build a RAG chain grounded in the given transcript.

    Parameters
    ----------
    transcript:
        Full transcript text to index.
    k:
        Number of chunks to retrieve per query.

    Returns
    -------
    A LangChain Runnable that accepts a question string and returns an answer.
    """
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=k)
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", "{question}")]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain built successfully.")
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """
    Invoke the RAG chain with a user question.

    Parameters
    ----------
    rag_chain:
        Chain returned by :func:`build_rag_chain`.
    question:
        The user's question text.

    Returns
    -------
    The LLM's answer as a string.
    """
    logger.info("RAG question: %s", question[:120])
    return rag_chain.invoke(question)
