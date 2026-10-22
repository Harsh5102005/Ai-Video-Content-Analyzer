# 🚀 How to Run — AI Video Content Analyzer

> Complete step-by-step guide to set up and run the project from scratch on **Windows**, **macOS**, or **Linux**.

---

## 📋 Table of Contents

1. [Prerequisites](#-1-prerequisites)
2. [Clone the Repository](#-2-clone-the-repository)
3. [Create a Virtual Environment](#-3-create-a-virtual-environment)
4. [Install Dependencies](#-4-install-dependencies)
5. [Get Your NVIDIA API Key](#-5-get-your-nvidia-api-key)
6. [Configure Environment Variables](#-6-configure-environment-variables)
7. [Run the Web App (Streamlit)](#-7-run-the-web-app-streamlit)
8. [Run via CLI (Terminal)](#-8-run-via-cli-terminal)
9. [Using the App](#-9-using-the-app)
10. [Troubleshooting](#-10-troubleshooting)
11. [Optional Configuration](#-11-optional-configuration)

---

## ✅ 1. Prerequisites

Make sure the following are installed on your system **before** starting:

### Python 3.10+

```bash
# Check your Python version
python --version
# or
python3 --version
```

> Download Python from [python.org](https://www.python.org/downloads/) if not installed.

---

### FFmpeg

FFmpeg is required for audio extraction and conversion.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install ffmpeg -y
```

**macOS (via Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
1. Download from [ffmpeg.org/download](https://ffmpeg.org/download.html)
2. Extract and add the `bin/` folder to your system `PATH`

**Verify FFmpeg is installed:**
```bash
ffmpeg -version
```

---

### Git

```bash
# Linux
sudo apt install git -y

# macOS
brew install git

# Windows — download from https://git-scm.com
```

---

## 📥 2. Clone the Repository

```bash
git clone https://github.com/harsh5102005/Ai-Video-Content-Analyzer.git
cd Ai-Video-Content-Analyzer
```

---

## 🐍 3. Create a Virtual Environment

A virtual environment keeps project dependencies isolated from your system Python.

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> ✅ You should see `(.venv)` at the start of your terminal prompt after activation.

---

## 📦 4. Install Dependencies

With your virtual environment **activated**, run:

```bash
pip install -r requirements.txt
```

> ⏳ This may take **5–10 minutes** on the first run — it downloads PyTorch, Whisper, ChromaDB, and other heavy packages.

**If you use `uv` (faster alternative to pip):**
```bash
uv pip install -r requirements.txt
```

**Verify key packages installed correctly:**
```bash
python -c "import whisper, streamlit, langchain_nvidia_ai_endpoints; print('All good!')"
```

---

## 🔑 5. Get Your NVIDIA API Key

1. Go to **[build.nvidia.com](https://build.nvidia.com)**
2. Create a **free account** (no credit card needed for trial credits)
3. Search for **`nemotron-3-ultra-550b`** or browse the catalog
4. Click the model → click **"Get API Key"**
5. Copy the key — it looks like: `nvapi-xxxxxxxxxxxxxxxxxxxx`

---

## ⚙️ 6. Configure Environment Variables

### Step 1 — Copy the example file

**Linux / macOS:**
```bash
cp .env.example .env
```

**Windows (Command Prompt):**
```cmd
copy .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

---

### Step 2 — Add your API key

Open `.env` in any text editor and replace the placeholder:

```env
# .env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **Never commit your `.env` file to Git.** It is already listed in `.gitignore`.

---

## 🌐 7. Run the Web App (Streamlit)

Make sure your virtual environment is **activated**, then:

```bash
streamlit run app.py
```

Streamlit will print a local URL like:

```
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser.

> 💡 The app will stay running until you press `Ctrl + C` in the terminal.

---

## 💻 8. Run via CLI (Terminal)

You can also run the full pipeline from the command line without opening a browser.

### Interactive mode (guided prompts):
```bash
python main.py
```

### With flags (non-interactive):
```bash
# Analyze a YouTube video
python main.py --source "https://youtube.com/watch?v=dQw4w9WgXcQ" --language english

# Analyze a local file
python main.py --source meeting.mp4 --language english

# Hindi/Hinglish video (auto-translated to English)
python main.py --source "https://youtube.com/watch?v=..." --language hindi

# Skip the interactive chat at the end
python main.py --source meeting.mp4 --no-chat
```

### CLI flags reference:

| Flag | Short | Description | Default |
|---|---|---|---|
| `--source` | `-s` | YouTube URL or local file path | *(prompted)* |
| `--language` | `-l` | `english` / `hindi` / `hinglish` | `english` |
| `--no-chat` | | Skip the interactive Q&A chat | `False` |

---

## 🎮 9. Using the App

### Web App Workflow

```
Sidebar                          Main area
───────                          ─────────
1. Choose source type            Landing page with instructions
   ├── YouTube URL  ──────────►  Paste a YouTube link
   └── Upload file  ──────────►  Upload MP4/MP3/WAV etc.

2. Select audio language
   (english / hindi / hinglish)

3. Click "🚀 Run Analysis"       Progress bar appears
                                 Whisper transcribes audio
                                 NVIDIA Nemotron analyses transcript

                                 ┌─────────────────────────────┐
                                 │  📋 Summary tab             │
                                 │  ├── Professional summary   │
                                 │  ├── ✅ Action Items        │
                                 │  ├── 🔑 Key Decisions       │
                                 │  └── ❓ Open Questions      │
                                 │                             │
                                 │  📝 Transcript tab          │
                                 │  ├── Search box             │
                                 │  └── Full transcript        │
                                 │                             │
                                 │  💬 Chat tab                │
                                 │  ├── Sample questions       │
                                 │  └── AI chat interface      │
                                 └─────────────────────────────┘

4. Export results
   ├── 📥 Download report (.md)
   └── 📄 Download transcript (.txt)
```

### Supported Input Formats

| Type | Supported formats |
|---|---|
| **YouTube** | Any public YouTube URL |
| **Video** | `.mp4` `.mov` `.mkv` `.webm` |
| **Audio** | `.mp3` `.wav` `.m4a` `.ogg` |

---

## 🛠️ 10. Troubleshooting

### ❌ `NVIDIA_API_KEY is not set`
Your `.env` file is missing or the key is wrong.
```bash
cat .env          # Check contents
# Should show: NVIDIA_API_KEY=nvapi-...
```

---

### ❌ `ffmpeg: command not found`
FFmpeg is not installed or not in PATH.
```bash
# Linux
sudo apt install ffmpeg -y

# macOS
brew install ffmpeg
```

---

### ❌ `ModuleNotFoundError: No module named 'whisper'`
Your virtual environment is not activated or dependencies aren't installed.
```bash
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

---

### ❌ Whisper is very slow / taking too long
You are running on CPU. To check:
```python
import torch
print(torch.cuda.is_available())   # True = GPU available, False = CPU only
```
- On CPU, a 10-minute video may take **5–15 minutes** to transcribe.
- To speed it up, use a smaller Whisper model:
  ```env
  # In your .env file
  WHISPER_MODEL=tiny
  ```
  Available sizes: `tiny` → `base` → `small` → `medium` → `large`

---

### ❌ YouTube download fails
- Make sure the video is **public** (not private or age-restricted)
- Update yt-dlp:
  ```bash
  pip install --upgrade yt-dlp
  ```

---

### ❌ `ChromaDB` or embedding errors
Delete the old vector database and retry:
```bash
rm -rf vector_db/
```

---

### ❌ Port 8501 already in use
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

---

## 🔧 11. Optional Configuration

You can customize behaviour by adding these to your `.env` file:

```env
# Which NVIDIA NIM model to use
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b

# Whisper model size (tiny is fastest, large is most accurate)
# Options: tiny | base | small | medium | large
WHISPER_MODEL=base

# HuggingFace embedding model for RAG
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Where downloaded YouTube audio is saved
DOWNLOAD_DIR=downloads

# Audio chunk duration in minutes (smaller = less RAM, more API calls)
CHUNK_MINUTES=10
```

---

## 📁 Project Structure (Quick Reference)

```
Ai-Video-Content-Analyzer/
├── app.py                    ← Streamlit web UI  (run this)
├── main.py                   ← CLI entry point
├── .env                      ← Your API keys (create from .env.example)
├── .env.example              ← Template for .env
├── requirements.txt          ← All Python dependencies
│
├── core/
│   ├── transcribe.py         ← Whisper speech-to-text
│   ├── summarize.py          ← LLM summarisation (map-reduce)
│   ├── extractor.py          ← Action items / decisions / questions
│   ├── rag_engine.py         ← RAG chat chain
│   └── vector_store.py       ← ChromaDB vector store
│
└── utils/
    └── audio_processing.py   ← Download · convert · chunk audio
```

---

> 💡 **Tip for your MCA presentation:** Use a short YouTube video (3–5 minutes) for a live demo — it runs faster and is easier to explain to the audience.
