# AI Video Content Analyzer 🎬

> **MCA Minor Project** — An end-to-end AI pipeline that converts YouTube videos and local meeting recordings into actionable intelligence: transcript, professional summary, action items, key decisions, open questions, and a transcript-grounded AI chat assistant.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **Local Transcription** | OpenAI Whisper runs on-device (GPU or CPU auto-detected) |
| 🌐 **Multi-language** | Hindi & Hinglish automatically translated to English |
| 📋 **Smart Summarisation** | Map-reduce summarisation for long transcripts |
| ✅ **Action Item Extraction** | Task · Owner · Deadline for every action item |
| 🔑 **Key Decision Extraction** | All decisions with context and attribution |
| ❓ **Open Question Detection** | Unresolved follow-ups captured automatically |
| 🤖 **RAG Chat** | Ask any question — answers grounded in the transcript |
| 📥 **Export** | Download Markdown report + raw transcript |
| 🔍 **Transcript Search** | In-app search with highlighted match context |
| 📺 **YouTube support** | Paste any public YouTube URL |
| 📁 **File upload** | MP4, MP3, WAV, M4A, MOV, MKV, WEBM, OGG |

---

## 🏗️ Architecture

```
Ai-Video-Content-Analyzer/
├── app.py                    # Streamlit web UI
├── main.py                   # CLI entry point (argparse)
├── core/
│   ├── __init__.py
│   ├── transcribe.py         # Whisper transcription (CUDA → CPU auto-fallback)
│   ├── summarize.py          # Map-reduce summarisation + title generation
│   ├── extractor.py          # Action items · decisions · open questions
│   ├── rag_engine.py         # LangChain RAG chat chain
│   └── vector_store.py       # Chroma vector store + HuggingFace embeddings
├── utils/
│   ├── __init__.py
│   └── audio_processing.py   # YT download · format conversion · chunking
├── requirements.txt
├── .env.example
└── .gitignore
```

### Pipeline flow

```
Source (URL / file)
        │
        ▼
utils/audio_processing.py  →  WAV chunks
        │
        ▼
core/transcribe.py  →  full transcript text
        │
        ├──► core/summarize.py       →  title + summary
        ├──► core/extractor.py       →  action items / decisions / questions
        └──► core/vector_store.py    →  Chroma embeddings
                    │
                    ▼
             core/rag_engine.py      →  RAG chat chain
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io) |
| **Speech-to-Text** | [OpenAI Whisper](https://github.com/openai/whisper) (local) |
| **LLM** | [NVIDIA NIM](https://build.nvidia.com) — `nvidia/nemotron-3-ultra-550b-a55b` |
| **Orchestration** | [LangChain](https://www.langchain.com) (LCEL) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Audio** | yt-dlp · pydub · FFmpeg |

---

## ⚙️ Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available in `PATH`  
  → [ffmpeg.org/download](https://ffmpeg.org/download.html)
- An **NVIDIA API key** → [build.nvidia.com](https://build.nvidia.com) (free account)
- *(Optional but recommended)* An NVIDIA GPU with CUDA for faster Whisper inference

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/harsh5102005/Ai-Video-Content-Analyzer.git
cd Ai-Video-Content-Analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Then open .env and set your MISTRAL_API_KEY
```

**`.env` file:**
```env
NVIDIA_API_KEY=nvapi-...your-key-here...

# Optional overrides
# WHISPER_MODEL=small        # tiny | base | small | medium | large
# NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
# CHUNK_MINUTES=10
```

---

## ▶️ Running the app

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### CLI mode

```bash
# Interactive prompts
python main.py

# Non-interactive flags
python main.py --source "https://youtube.com/watch?v=..." --language english
python main.py --source meeting.mp4 --language hindi --no-chat
```

CLI flags:

| Flag | Description |
|---|---|
| `--source` / `-s` | YouTube URL or local file path |
| `--language` / `-l` | `english` · `hindi` · `hinglish` (default: `english`) |
| `--no-chat` | Skip the interactive chat session |

---

## 📝 Notes

- `downloads/` — generated audio files (git-ignored)
- `vector_db/` — local Chroma vector store (git-ignored)
- `.env` — API keys (git-ignored)
- Large audio/video files should never be committed to GitHub

---

## 📄 Resume Description

**AI Video Content Analyzer** — Built an end-to-end Streamlit + CLI application that converts YouTube videos and meeting recordings into searchable meeting intelligence. Implemented local Whisper speech-to-text with automatic CUDA/CPU device detection, Hindi/Hinglish-to-English translation, map-reduce LLM summarisation, structured extraction of action items and decisions, and a LangChain/Chroma RAG chatbot for transcript-grounded Q&A using NVIDIA NIM (Nemotron-3-Ultra-550B).
