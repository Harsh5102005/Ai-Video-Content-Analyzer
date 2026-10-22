"""
main.py
-------
CLI entry point for the AI Video Content Analyzer pipeline.

Usage:
    python main.py
    python main.py --source <url_or_path> --language english
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarize import generate_title, summarize
from core.transcribe import transcribe_all
from utils.audio_processing import cleanup_audio_files, process_input

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "=" * 65


def run_pipeline(source: str, language: str = "english") -> dict:
    """Execute the full analysis pipeline and return results."""
    logger.info("Starting AI Video Content Analyzer pipeline …")

    logger.info("Processing source: %s", source)
    chunks = process_input(source)

    logger.info("Transcribing %d chunk(s) …", len(chunks))
    translate = language.lower() in {"hindi", "hinglish"}
    transcript = transcribe_all(chunks, translate=translate)

    # Clean up temp audio chunks
    cleanup_audio_files(chunks)

    logger.info("Raw transcript length: %d characters", len(transcript))

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    open_questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title.strip(),
        "transcript": transcript,
        "summary": summary,
        "action_item": action_item,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


def _print_results(result: dict) -> None:
    print(f"\n{SEP}")
    print(f"📌  Title:          {result['title']}")
    print(SEP)
    print(f"\n📋  Summary:\n{result['summary']}")
    print(f"\n✅  Action Items:\n{result['action_item']}")
    print(f"\n🔑  Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓  Open Questions:\n{result['open_questions']}")
    print(f"\n{SEP}\n")


def _interactive_chat(rag_chain) -> None:
    print("💬  Chat with your meeting transcript (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("👋  Goodbye!")
            break
        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\n🤖  Assistant: {answer}\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Video Content Analyzer — CLI mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        help="YouTube URL or local audio/video file path.",
    )
    parser.add_argument(
        "--language", "-l",
        default="english",
        choices=["english", "hindi", "hinglish"],
        help="Audio language (default: english).",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Skip the interactive chat session after analysis.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    source = args.source
    if not source:
        source = input("Enter YouTube URL or local file path: ").strip()
    if not source:
        print("❌  No source provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    language = args.language

    try:
        result = run_pipeline(source, language)
    except EnvironmentError as exc:
        print(f"\n❌  Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed")
        print(f"\n❌  Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_results(result)

    if not args.no_chat:
        _interactive_chat(result["rag_chain"])


if __name__ == "__main__":
    main()