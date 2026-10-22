"""
core/summarize.py
-----------------
Map-reduce summarisation and title generation via Mistral AI + LangChain.
"""

import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
_TITLE_PREVIEW_CHARS = 3000  # feed only the first N chars for title generation


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


def _split_transcript(
    transcript: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(transcript)


_MAP_SYSTEM = (
    "You are a professional meeting analyst. "
    "Concisely summarise the following portion of a meeting transcript, "
    "preserving all key points, names, and decisions."
)

_REDUCE_SYSTEM = (
    "You are an expert meeting summariser. "
    "Combine the partial summaries below into ONE final, professional meeting summary. "
    "Use clear bullet points grouped by theme. Keep it informative and concise."
)

_TITLE_SYSTEM = (
    "Based on the meeting transcript excerpt, generate a short, professional meeting title. "
    "Maximum 8 words. Return ONLY the title — nothing else."
)


def summarize(transcript: str) -> str:
    """
    Produce a professional summary of the full transcript using map-reduce.

    Long transcripts are split into chunks, each chunk is summarised
    individually (map), then all partial summaries are combined (reduce).
    """
    llm = _get_llm()

    # ── MAP phase ────────────────────────────────────────────────────────────
    map_prompt = ChatPromptTemplate.from_messages(
        [("system", _MAP_SYSTEM), ("human", "{text}")]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = _split_transcript(transcript)
    logger.info("Summarising %d chunk(s) …", len(chunks))
    partial_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    # ── REDUCE phase ─────────────────────────────────────────────────────────
    combined = "\n\n".join(partial_summaries)
    reduce_prompt = ChatPromptTemplate.from_messages(
        [("system", _REDUCE_SYSTEM), ("human", "{text}")]
    )
    reduce_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | reduce_prompt
        | llm
        | StrOutputParser()
    )
    return reduce_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    """Generate a short meeting title from the beginning of the transcript."""
    llm = _get_llm(temperature=0.2)
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [("system", _TITLE_SYSTEM), ("human", "{text}")]
        )
        | llm
        | StrOutputParser()
    )
    preview = transcript[:_TITLE_PREVIEW_CHARS]
    return title_chain.invoke(preview).strip()
