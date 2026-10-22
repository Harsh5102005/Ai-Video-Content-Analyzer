"""
core/transcribe.py
------------------
Whisper-based speech-to-text with automatic device fallback (CUDA → CPU).
"""

import logging
import os

import torch
import whisper

logger = logging.getLogger(__name__)

_model: whisper.Whisper | None = None


def _detect_device() -> str:
    """Return 'cuda' if an NVIDIA GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load (and cache) the Whisper model.

    The model is loaded once per process and reused on subsequent calls.
    Model size can be overridden via the WHISPER_MODEL env-var
    (e.g. ``WHISPER_MODEL=small``).
    """
    global _model
    if _model is None:
        size = os.getenv("WHISPER_MODEL", model_size)
        device = _detect_device()
        logger.info("Loading Whisper model '%s' on device '%s'", size, device)
        _model = whisper.load_model(size, device=device)
        logger.info("Whisper model loaded successfully.")
    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    """
    Transcribe (or translate) a single audio chunk.

    Parameters
    ----------
    chunk_path:
        Filesystem path to the WAV chunk.
    translate:
        If ``True`` the task is set to ``"translate"`` (output in English).
    """
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result["text"]


def transcribe_all(
    chunks: list[str],
    translate: bool = False,
    progress_cb=None,
) -> str:
    """
    Transcribe all audio chunks and return the concatenated transcript.

    Parameters
    ----------
    chunks:
        List of WAV file paths produced by ``chunk_audio``.
    translate:
        Passed through to :func:`transcribe_chunk`.
    progress_cb:
        Optional callable(current, total) called after each chunk.
    """
    parts: list[str] = []
    total = len(chunks)
    for i, chunk_path in enumerate(chunks, start=1):
        logger.info("Transcribing chunk %d / %d …", i, total)
        text = transcribe_chunk(chunk_path, translate=translate)
        parts.append(text)
        if progress_cb:
            progress_cb(i, total)
    return " ".join(parts)
