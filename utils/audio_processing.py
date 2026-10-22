"""
utils/audio_processing.py
--------------------------
Audio acquisition, format conversion, and chunking utilities.
"""

import logging
import os
import tempfile

from pydub import AudioSegment
import yt_dlp

logger = logging.getLogger(__name__)

DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Chunking defaults — can be overridden via env-vars
_DEFAULT_CHUNK_MINUTES: int = int(os.getenv("CHUNK_MINUTES", "10"))
_TARGET_SAMPLE_RATE: int = 16_000  # Whisper expects 16 kHz


def download_yt_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to a 16 kHz WAV file.

    Parameters
    ----------
    url:
        Public YouTube video URL.

    Returns
    -------
    Absolute path to the downloaded WAV file.
    """
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "quiet": True,          # suppress yt-dlp stdout noise
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_path = ydl.prepare_filename(info)

    wav_path = os.path.splitext(raw_path)[0] + ".wav"
    logger.info("Downloaded audio to: %s", wav_path)
    return wav_path


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to a mono 16 kHz WAV suitable for Whisper.

    The output file is placed next to the input file with a ``_converted``
    suffix so the original is preserved.

    Parameters
    ----------
    input_path:
        Path to the source audio/video file.

    Returns
    -------
    Path to the converted WAV file.
    """
    base, _ = os.path.splitext(input_path)
    output_path = base + "_converted.wav"
    logger.info("Converting '%s' → '%s' …", input_path, output_path)
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(_TARGET_SAMPLE_RATE)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = _DEFAULT_CHUNK_MINUTES,
) -> list[str]:
    """
    Split a WAV file into equal-length chunks for parallel / sequential transcription.

    Chunk files are saved alongside the source WAV and named ``<base>_chunkN.wav``.

    Parameters
    ----------
    wav_path:
        Path to the source WAV file.
    chunk_minutes:
        Duration of each chunk in minutes (default: 10).

    Returns
    -------
    Ordered list of chunk file paths.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1_000
    base = os.path.splitext(wav_path)[0]
    chunks: list[str] = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base}_chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        logger.debug("Saved chunk %d → %s", i, chunk_path)

    logger.info("Audio split into %d chunk(s).", len(chunks))
    return chunks


def cleanup_audio_files(paths: list[str]) -> None:
    """
    Remove temporary audio chunk files from disk.

    Silently ignores files that no longer exist.

    Parameters
    ----------
    paths:
        List of file paths to delete.
    """
    for path in paths:
        try:
            os.remove(path)
            logger.debug("Removed temp file: %s", path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove '%s': %s", path, exc)


def process_input(source: str) -> list[str]:
    """
    Detect the source type, acquire audio, and return a list of WAV chunk paths.

    Parameters
    ----------
    source:
        Either a YouTube URL (``http://`` / ``https://``) or a local file path.

    Returns
    -------
    Ordered list of WAV chunk file paths ready for transcription.
    """
    if source.startswith(("http://", "https://")):
        logger.info("Detected YouTube URL — downloading audio …")
        wav_path = download_yt_audio(source)
    else:
        logger.info("Detected local file — converting to WAV …")
        wav_path = convert_to_wav(source)

    chunks = chunk_audio(wav_path)
    return chunks
