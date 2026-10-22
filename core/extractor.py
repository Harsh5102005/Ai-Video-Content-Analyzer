"""
core/extractor.py
-----------------
LLM-based extraction of action items, key decisions, and open questions
from a meeting transcript using Mistral AI + LangChain.
"""

import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)

_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")


def _get_llm(temperature: float = 0.2) -> ChatNVIDIA:
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


def _build_chain(system_prompt: str):
    """Build a simple text→LLM→text chain with the given system prompt."""
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text}")]
    )
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


_ACTION_ITEMS_PROMPT = (
    "You are an expert meeting analyst. "
    "From the meeting transcript below, extract ALL action items. "
    "For each item provide:\n"
    "  - **Task**: A clear description of what needs to be done\n"
    "  - **Owner**: Who is responsible (use 'Unassigned' if not mentioned)\n"
    "  - **Deadline**: The due date or timeline (use 'Not specified' if absent)\n\n"
    "Format as a numbered list. "
    "If no action items are found, respond with: 'No action items identified.'"
)

_DECISIONS_PROMPT = (
    "You are an expert meeting analyst. "
    "From the meeting transcript below, extract ALL key decisions that were made. "
    "For each decision include what was decided and, if mentioned, who made the decision.\n"
    "Format as a numbered list. "
    "If no decisions are found, respond with: 'No key decisions identified.'"
)

_QUESTIONS_PROMPT = (
    "You are an expert meeting analyst. "
    "From the meeting transcript below, extract ALL unresolved questions, "
    "open issues, or topics that require follow-up. "
    "For each, briefly describe the question and, if available, who raised it.\n"
    "Format as a numbered list. "
    "If none are found, respond with: 'No open questions identified.'"
)


def extract_action_items(transcript: str) -> str:
    """Extract action items from the transcript."""
    logger.info("Extracting action items …")
    return _build_chain(_ACTION_ITEMS_PROMPT).invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from the transcript."""
    logger.info("Extracting key decisions …")
    return _build_chain(_DECISIONS_PROMPT).invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Extract open / unresolved questions from the transcript."""
    logger.info("Extracting open questions …")
    return _build_chain(_QUESTIONS_PROMPT).invoke(transcript)