"""
app.py
------
AI Video Content Analyzer — Streamlit UI.

Run with:  streamlit run app.py
"""

import logging
import os
import re
import tempfile
from datetime import datetime
from html import escape
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.summarize import generate_title, summarize
from core.transcribe import transcribe_all
from utils.audio_processing import cleanup_audio_files, process_input

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Content Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Global background ── */
.stApp {
    background: #0a0f1e;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,.15);
}

[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: .02em;
    transition: opacity .2s, transform .15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: .9;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(99,102,241,.3) !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Hero banner ── */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.4rem 2.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,.25);
    margin-bottom: 1.6rem;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 80% 20%, rgba(99,102,241,.20) 0%, transparent 50%),
        radial-gradient(circle at 20% 80%, rgba(139,92,246,.15) 0%, transparent 50%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    padding: .28rem .75rem;
    background: rgba(99,102,241,.18);
    border: 1px solid rgba(99,102,241,.4);
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: .9rem;
}
.hero h1 {
    margin: 0;
    font-size: clamp(1.8rem, 3.5vw, 3.2rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    max-width: 700px;
    margin: .9rem 0 0;
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.65;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin: 1.4rem 0 1.6rem;
}
.stat-card {
    position: relative;
    overflow: hidden;
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    transition: border-color .2s, transform .2s;
}
.stat-card:hover {
    border-color: rgba(99,102,241,.5);
    transform: translateY(-2px);
}
.stat-label {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #64748b;
}
.stat-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #e2e8f0;
    margin-top: .25rem;
}
.stat-icon {
    position: absolute;
    right: 1.1rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.7rem;
    opacity: .2;
}

/* ── Feature card (landing) ── */
.feature-card {
    padding: 1.5rem 1.6rem;
    border-radius: 18px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    height: 100%;
    transition: border-color .25s, transform .2s;
}
.feature-card:hover {
    border-color: rgba(99,102,241,.45);
    transform: translateY(-3px);
}
.feature-card .step-num {
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: .4rem;
}
.feature-card h3 {
    margin: 0 0 .5rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.feature-card p {
    margin: 0;
    color: #64748b;
    font-size: .92rem;
    line-height: 1.6;
}
.feature-emoji {
    font-size: 2rem;
    margin-bottom: .7rem;
}

/* ── Export button strip ── */
.stDownloadButton > button {
    background: rgba(99,102,241,.12) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: background .2s, transform .15s !important;
}
.stDownloadButton > button:hover {
    background: rgba(99,102,241,.25) !important;
    transform: translateY(-1px);
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,.04) !important;
    border-radius: 12px !important;
    gap: .35rem !important;
    padding: .3rem .4rem !important;
}
[data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: .92rem !important;
    transition: background .2s, color .2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: .9rem !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,.07) !important;
}

/* ── Sample question buttons ── */
.sample-btn > button {
    background: rgba(99,102,241,.08) !important;
    border: 1px solid rgba(99,102,241,.25) !important;
    color: #a5b4fc !important;
    border-radius: 9px !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    transition: background .2s !important;
}
.sample-btn > button:hover {
    background: rgba(99,102,241,.2) !important;
}

/* ── Responsive ── */
@media (max-width: 900px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 560px) {
    .stat-grid { grid-template-columns: 1fr; }
    .hero { padding: 1.5rem 1.4rem; }
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(value: str, fallback: str = "ai_video_report") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value or fallback).strip("_")
    return (cleaned[:80] or fallback).lower()


def transcript_stats(transcript: str) -> dict:
    words = re.findall(r"\b\w+\b", transcript or "")
    minutes = max(1, round(len(words) / 150)) if words else 0
    sentences = len(re.findall(r"[.!?]+", transcript or ""))
    return {
        "words": len(words),
        "characters": len(transcript or ""),
        "reading_time": minutes,
        "sentences": sentences,
    }


def build_markdown_report(result: dict) -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# {result["title"]}

> Generated by AI Video Content Analyzer on {created_at}

---

## 📋 Summary

{result["summary"]}

---

## ✅ Action Items

{result["action_item"]}

---

## 🔑 Key Decisions

{result["key_decisions"]}

---

## ❓ Open Questions

{result["open_questions"]}

---

## 📝 Full Transcript

{result["transcript"]}
"""


def build_txt_transcript(result: dict) -> str:
    return result["transcript"]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(source: str, language: str = "english", progress_cb=None):
    """Execute the full analysis pipeline and return a result dict."""

    def step(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    step("⚙️  Preparing audio source …")
    chunks = process_input(source)

    step("🎙️  Transcribing audio with Whisper …")
    translate = language.lower() in {"hindi", "hinglish"}
    transcript = transcribe_all(chunks, translate=translate)

    # Clean up temp chunks after transcription
    cleanup_audio_files(chunks)

    step("🏷️  Generating meeting title …")
    title = generate_title(transcript)

    step("📋  Writing professional summary …")
    summary = summarize(transcript)

    step("✅  Extracting action items …")
    action_item = extract_action_items(transcript)

    step("🔑  Identifying key decisions …")
    key_decisions = extract_key_decisions(transcript)

    step("❓  Finding open questions …")
    open_questions = extract_questions(transcript)

    step("🧠  Building transcript chat index …")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title.strip(),
        "transcript": transcript,
        "summary": summary,
        "action_item": action_item,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "language": language,
    }


# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "result": None,
        "chat_history": [],
        "processing": False,
        "status_history": [],
        "pending_question": None,
        "error_message": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.3rem;">
            <span style="font-size:1.7rem;">🎬</span>
            <span style="font-size:1.1rem;font-weight:800;color:#e2e8f0;">AI Video Analyzer</span>
        </div>
        <p style="color:#64748b;font-size:.83rem;margin:0 0 .5rem;">
            Transcribe · Summarize · Chat
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    input_mode = st.radio(
        "📂 Source type",
        ["YouTube URL", "Upload file"],
        index=0,
        horizontal=True,
    )

    source: str | None = None
    tmp_path_to_cleanup: str | None = None

    if input_mode == "YouTube URL":
        raw_url = st.text_input(
            "Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any public YouTube video URL.",
        ).strip()
        source = raw_url or None
    else:
        uploaded = st.file_uploader(
            "Audio / Video file",
            type=["mp4", "mp3", "wav", "m4a", "mov", "mkv", "webm", "ogg"],
            help="Converted to 16 kHz WAV before transcription.",
        )
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            source = tmp.name
            tmp_path_to_cleanup = tmp.name
            st.caption(f"📎 {uploaded.name}")

    language = st.selectbox(
        "🌐 Audio language",
        ["english", "hindi", "hinglish"],
        index=0,
        help=(
            "Hindi and Hinglish are automatically translated to English "
            "during transcription using Whisper's translate mode."
        ),
    )

    st.divider()

    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing or not source,
    )

    if st.session_state.result:
        if st.button("🔄 Start Over", use_container_width=True):
            for key in ("result", "chat_history", "status_history", "pending_question", "error_message"):
                st.session_state[key] = [] if key in ("chat_history", "status_history") else None
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='color:#475569;font-size:.78rem;'>"
        "💡 <b>Tip:</b> Use shorter clips for faster testing. "
        "Whisper runs locally and can take a few minutes on CPU."
        "</p>",
        unsafe_allow_html=True,
    )


# ── Pipeline execution ────────────────────────────────────────────────────────

if run_clicked and source:
    st.session_state.processing = True
    st.session_state.chat_history = []
    st.session_state.status_history = []
    st.session_state.error_message = None

    TOTAL_STEPS = 8
    step_count = {"n": 0}
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Starting analysis …")

    def _progress_cb(message: str):
        step_count["n"] += 1
        pct = min(int(step_count["n"] / TOTAL_STEPS * 100), 95)
        st.session_state.status_history.append(message)
        progress_bar.progress(pct, text=message)
        status_placeholder.info(message)

    try:
        with st.spinner("Processing your video — this may take a few minutes …"):
            result = run_pipeline(source, language, progress_cb=_progress_cb)
        st.session_state.result = result
        progress_bar.progress(100, text="✅ Analysis complete!")
        status_placeholder.success("Analysis complete! Scroll down to explore your results.")
    except EnvironmentError as env_err:
        st.session_state.error_message = (
            f"**Configuration error:** {env_err}\n\n"
            "Please check your `.env` file and make sure `MISTRAL_API_KEY` is set."
        )
        status_placeholder.empty()
        progress_bar.empty()
    except Exception as exc:
        logger.exception("Pipeline failed")
        st.session_state.error_message = f"**Pipeline error:** {exc}"
        status_placeholder.empty()
        progress_bar.empty()
    finally:
        st.session_state.processing = False
        # Clean up uploaded temp file
        if tmp_path_to_cleanup and os.path.exists(tmp_path_to_cleanup):
            try:
                os.remove(tmp_path_to_cleanup)
            except OSError:
                pass

# Surface errors prominently
if st.session_state.get("error_message"):
    st.error(st.session_state.error_message)

# ── Main display ──────────────────────────────────────────────────────────────

result = st.session_state.result

if not result:
    # ── Landing / welcome screen ──────────────────────────────────────────────
    st.markdown(
        """
        <section class="hero">
            <div class="hero-badge">✦ MCA Minor Project</div>
            <h1>Turn videos into<br>actionable intelligence.</h1>
            <p>
                Upload a recording or paste a YouTube URL to get a full English transcript,
                professional summary, action items, key decisions, open questions, and an
                AI chat assistant grounded in the transcript — all in one click.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("🎯", "STEP 1", "Add a source", "Paste a YouTube URL or upload an audio/video file from the sidebar."),
        ("⚡", "STEP 2", "Run analysis", "The app transcribes with Whisper, summarises, extracts insights, and builds a chat index."),
        ("📤", "STEP 3", "Explore & export", "Download a Markdown report, search the transcript, or ask questions in the AI chat."),
    ]
    for col, (emoji, step, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""<div class="feature-card">
                    <div class="feature-emoji">{emoji}</div>
                    <div class="step-num">{step}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    if st.session_state.status_history:
        with st.expander("📋 Last run — processing log"):
            for i, item in enumerate(st.session_state.status_history, 1):
                st.write(f"{i}. {item}")

else:
    # ── Results screen ────────────────────────────────────────────────────────
    stats = transcript_stats(result["transcript"])
    md_report = build_markdown_report(result)
    filename_base = safe_filename(result["title"])
    title_html = escape(result["title"])

    # Hero banner (results)
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-badge">✅ Analysis ready · {result.get("created_at", "")}</div>
            <h1>{title_html}</h1>
            <p>Review the summary below, explore the transcript, export your report, or ask the AI assistant questions about the video.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Stat cards
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Words</div>
                <div class="stat-value">{stats["words"]:,}</div>
                <div class="stat-icon">📝</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sentences</div>
                <div class="stat-value">{stats["sentences"]:,}</div>
                <div class="stat-icon">💬</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Reading time</div>
                <div class="stat-value">{stats["reading_time"]} min</div>
                <div class="stat-icon">⏱️</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Language</div>
                <div class="stat-value">{result.get("language", "english").title()}</div>
                <div class="stat-icon">🌐</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Export row
    exp1, exp2, exp3 = st.columns([1, 1, 2])
    with exp1:
        st.download_button(
            "📥 Download report (.md)",
            data=md_report,
            file_name=f"{filename_base}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "📄 Download transcript (.txt)",
            data=result["transcript"],
            file_name=f"{filename_base}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with exp3:
        if st.session_state.status_history:
            with st.expander("🕐 Processing timeline"):
                for i, item in enumerate(st.session_state.status_history, 1):
                    st.write(f"{i}. {item}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_summary, tab_transcript, tab_chat = st.tabs(
        ["📋  Summary", "📝  Transcript", "💬  Chat"]
    )

    # ── Summary tab ───────────────────────────────────────────────────────────
    with tab_summary:
        st.subheader("Professional Meeting Summary")
        st.markdown(result["summary"])

        st.divider()
        ins1, ins2, ins3 = st.columns(3)
        with ins1:
            with st.container(border=True):
                st.markdown("#### ✅ Action Items")
                st.markdown(result["action_item"])
        with ins2:
            with st.container(border=True):
                st.markdown("#### 🔑 Key Decisions")
                st.markdown(result["key_decisions"])
        with ins3:
            with st.container(border=True):
                st.markdown("#### ❓ Open Questions")
                st.markdown(result["open_questions"])

    # ── Transcript tab ────────────────────────────────────────────────────────
    with tab_transcript:
        search_term = st.text_input(
            "🔍 Search transcript",
            placeholder="Type a word or phrase to find …",
        )
        transcript_text = result["transcript"]

        if search_term:
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            matches = list(pattern.finditer(transcript_text))
            if matches:
                st.success(f"Found **{len(matches)}** match(es).")
                first = matches[0].start()
                ctx_start = max(0, first - 300)
                ctx_end = min(len(transcript_text), first + 600)
                # Highlight the match inside the context
                ctx = transcript_text[ctx_start:ctx_end]
                highlighted = pattern.sub(
                    lambda m: f"**{m.group()}**", ctx
                )
                st.markdown(
                    f"**First match context:**\n\n…{highlighted}…"
                )
            else:
                st.warning(f'No matches found for "{search_term}".')

        st.text_area(
            "Full transcript",
            transcript_text,
            height=550,
            label_visibility="collapsed",
        )

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("💬 Chat with the Transcript")
        st.caption(
            "Ask any question — answers are grounded in the indexed transcript context."
        )

        # Sample question buttons
        sample_questions = [
            "What are the main takeaways?",
            "What action items were assigned?",
            "What risks or concerns were raised?",
            "Who were the key speakers?",
        ]
        btn_cols = st.columns(len(sample_questions))
        for col, sq in zip(btn_cols, sample_questions):
            with col:
                if st.button(sq, use_container_width=True, key=f"sq_{sq[:15]}"):
                    st.session_state.pending_question = sq
                    st.rerun()

        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.pending_question = None
                st.rerun()

        st.divider()

        # Render chat history
        for q_text, a_text in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q_text)
            with st.chat_message("assistant"):
                st.write(a_text)

        # Accept new question
        question = (
            st.session_state.pending_question
            or st.chat_input("Ask something about this video …")
        )
        st.session_state.pending_question = None

        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript …"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as exc:
                        logger.exception("RAG error")
                        answer = f"⚠️ Error generating answer: {exc}"
                st.write(answer)
            st.session_state.chat_history.append((question, answer))
