# RAG + Voice Study Assistant

A study assistant that answers questions from **your own CS notes and past exam papers**,
cites the page it got each fact from, solves problems step by step, marks your answers
against the answer key, and can be used by voice.

```
$ study ask "What are the four conditions for deadlock?"
  -> search_notes(query='conditions for deadlock')
Deadlock needs all four Coffman conditions to hold at once [operating_systems.md p.2]:
1. Mutual exclusion ...
```

**Tech stack:** Python · Claude API (tool use, structured outputs) · PostgreSQL + pgvector ·
sentence-transformers (local embeddings) · faster-whisper (speech-to-text) · pyttsx3 (text-to-speech)

---

## Contents

1. [How it works](#1-how-it-works)
2. [What you need](#2-what-you-need)
3. [Setup, step by step](#3-setup-step-by-step)
4. [Try it with the sample data](#4-try-it-with-the-sample-data)
5. [Voice mode](#5-voice-mode)
6. [Use your own notes](#6-use-your-own-notes)
7. [Run the tests (no API key needed)](#7-run-the-tests-no-api-key-needed)
8. [Configuration](#8-configuration)
9. [Troubleshooting](#9-troubleshooting)
10. [Project structure](#10-project-structure)

---

## 1. How it works

```
            ┌──────────────── ingest ─────────────────┐
 PDF/MD ──► │ text per page → chunks (≤1000 chars,     │──► PostgreSQL + pgvector
            │ never crossing a page) → MiniLM (384-d)  │    documents(kind: notes|exam|answer_key)
            └──────────────────────────────────────────┘    chunks(page, content, embedding)
                                                                    ▲
 mic ─► Whisper STT ─┐                                              │ cosine similarity search
                     ▼                                              │
 your question ──► agent loop (Claude) ──tool call──► search_notes ─┘
                     ▲      │                     ├─► solve_step_by_step ─► Claude: numbered steps
                     │      └──tool result────────┴─► check_answer ──────► Claude: grade vs. answer key
                     ▼
              cited answer ──► (voice mode) "[x.pdf p.3]" → "from x, page 3" ──► speaker
```

1. **Ingest:** each file is read page by page and split into chunks. Every chunk keeps
   its file name and page number. Each chunk is converted to a 384-number vector (an
   "embedding") by a small model that runs on your computer. The vectors are stored in
   PostgreSQL using the pgvector extension.
2. **Ask:** your question goes to Claude together with three tools. Claude decides which
   tool to call:

   | Tool | When Claude uses it | What it does |
   |---|---|---|
   | `search_notes` | "What is…", "Explain…" | finds the most similar chunks and returns them with `[file p.N]` citations |
   | `solve_step_by_step` | "Solve…", "Compute…", "Trace…" | gets related notes and past solutions, then returns a numbered worked solution |
   | `check_answer` | "Check my answer…", "Mark this…" | finds the matching answer-key passage and returns a verdict, score out of 10, feedback and missed points. **If no key matches, it refuses to grade.** |

3. Claude writes the final answer and cites the pages it used. In voice mode the answer is
   also read aloud.

---

## 2. What you need

| Requirement | Why | How to get it |
|---|---|---|
| **Python 3.10 or newer** | runs the app | [python.org/downloads](https://www.python.org/downloads/). Check with `python --version` (or `python3 --version`). |
| **Docker Desktop** *(recommended)* | runs PostgreSQL + pgvector with one command | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/). Check with `docker --version`. Without Docker, see [Postgres without Docker](#postgres-without-docker). |
| **Anthropic API key** | lets the agent call Claude | Create one at [console.anthropic.com](https://console.anthropic.com/) → *API Keys*. Using the API costs money; the sample questions below cost a few cents. |
| **Git** | to download the code | [git-scm.com](https://git-scm.com/) |
| **About 3 GB of free disk space** | PyTorch, the embedding model (~90 MB) and the Whisper model (~150 MB) | |
| Microphone and speakers *(optional)* | only for voice mode | |

An internet connection is needed the first time, to download Python packages and the models.

---

## 3. Setup, step by step

Run all commands from a terminal: Terminal on macOS/Linux, PowerShell on Windows.

### Step 1: Get the code

```bash
git clone https://github.com/shaikabdul185-arch/robotics-lab.git
cd robotics-lab/study-assistant
```

> **Important:** run every `study ...` command from inside the `study-assistant` folder.
> That's where your `.env` file is read from.

### Step 2: Create a virtual environment

This keeps the project's packages separate from the rest of your system.

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell says running scripts is disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Your prompt should now start with `(.venv)`. **Activate it again in every new terminal**
before using `study`.

### Step 3: Install the packages

```bash
pip install --upgrade pip

# Linux only, optional: install the CPU-only PyTorch first. It's much smaller than the default GPU build.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the app. "voice" adds speech support and "dev" adds pytest.
pip install -e ".[voice,dev]"
```

To skip voice mode, use `pip install -e ".[dev]"` instead.

Check that it installed:
```bash
study --help
```

### Step 4: Start the database

Make sure Docker Desktop is running, then:

```bash
docker compose up -d
```

This starts PostgreSQL 16 with pgvector on `localhost:5432` (user `study`, password `study`,
database `study`). The data is kept in a Docker volume, so it survives restarts.

Check it's running:
```bash
docker compose ps        # STATUS should say "running" / "Up"
```

### Step 5: Add your API key

macOS / Linux:
```bash
cp .env.example .env
```
Windows:
```powershell
copy .env.example .env
```

Open `.env` in any text editor and paste your key after `ANTHROPIC_API_KEY=`:
```
ANTHROPIC_API_KEY=sk-ant-...your key...
DATABASE_URL=postgresql://study:study@localhost:5432/study
```

> `.env` is in `.gitignore`. Never commit your API key.

### Step 6: Create the tables

```bash
study init-db
```
Expected output:
```
Schema ready.
```

---

## 4. Try it with the sample data

`sample_data/` contains a small example course:

```
sample_data/
  notes/data_structures.md      BSTs (p.1), B-trees (p.2), hash tables (p.3)
  notes/operating_systems.md    processes (p.1), deadlock (p.2), CPU scheduling (p.3)
  exams/midterm_2025.md         a midterm with questions Q1–Q3
  keys/midterm_2025_key.md      the marking scheme for Q1–Q3
```

In the Markdown files, a line containing only `---` marks a new page. PDFs use their real pages.

### 4.1 Ingest

```bash
study ingest sample_data/
```
Expected output. The first run downloads the embedding model, which takes about a minute:
```
 ingested  sample_data/exams/midterm_2025.md (2 chunks, exam)
 ingested  sample_data/keys/midterm_2025_key.md (2 chunks, answer_key)
 ingested  sample_data/notes/data_structures.md (3 chunks, notes)
 ingested  sample_data/notes/operating_systems.md (3 chunks, notes)
```
Run it again and every file shows `unchanged`, because unchanged files are skipped.

```bash
study stats
```
```
answer_key     1 documents       2 chunks
exam           1 documents       2 chunks
notes          2 documents       6 chunks
```

### 4.2 Check retrieval (no API key used)

`search` shows exactly what the agent would retrieve, without calling Claude:

```bash
study search "how do I prevent deadlock" -k 2
```
You should get `[operating_systems.md p.2]` first. The similarity numbers will vary.
```
[operating_systems.md p.2] (notes, similarity 0.6x)
## Deadlock Four conditions must all hold for deadlock (Coffman conditions): ...
```

### 4.3 Ask questions and try all three tools

The line starting with `->` shows which tool Claude chose.

**Search the notes:**
```bash
study ask "Why do databases use B-trees instead of binary search trees?"
```
Expect `-> search_notes(...)` and an answer citing `[data_structures.md p.2]` (and probably p.1).

**Solve step by step:**
```bash
study ask "Solve: three processes arrive at time 0 with bursts 24, 3 and 3. What is the average waiting time under FCFS and under SJF?"
```
Expect `-> solve_step_by_step(...)`, numbered steps, and the final answer **FCFS = 17, SJF = 3**.

**Check an answer against the key:**
```bash
study ask "Check my answer to midterm Q1: the conditions are mutual exclusion, hold and wait, and no preemption. To prevent deadlock, use locks."
```
Expect `-> check_answer(...)`, a verdict of **partially_correct**, and feedback saying you
missed *circular wait* and that "use locks" is not a valid prevention method. The grade
should cite `[midterm_2025_key.md p.1]`.

**Ask about something that isn't in the notes:**
```bash
study ask "Check my answer to the question about red-black tree rotations: you rotate left."
```
There is no answer key for this, so the assistant should say it can't grade it, not make up a score.

### 4.4 Multi-turn chat

```bash
study chat
```
```
you> What is a B-tree?
assistant> ... [data_structures.md p.2]
you> How is a B+ tree different?
assistant> ...                       (it remembers the previous question)
you> /reset                          (start a new conversation)
you> /quit
```

---

## 5. Voice mode

Voice mode uses the same agent: your speech becomes the question, and the answer is printed
**and** read aloud. Citations are spoken as "from operating systems, page 2".

### Extra system packages

| OS | Microphone (PortAudio) | Text-to-speech engine |
|---|---|---|
| **Windows** | included with `sounddevice` | built in (SAPI5) |
| **macOS** | included with `sounddevice` | built in (NSSpeechSynthesizer) |
| **Ubuntu/Debian** | `sudo apt install libportaudio2` | `sudo apt install espeak-ng` |

### Use it

```bash
study voice
```
1. Press **Enter** and ask your question out loud.
2. Press **Enter** again to stop recording.
3. The transcript, the answer and the tool calls are printed, then the answer is spoken.
4. Press **Ctrl+C** to quit.

The first run downloads the Whisper `base.en` model (about 150 MB).

### Test it without a microphone or speakers

Record a question as a `.wav` file (with your phone, Audacity or Voice Recorder), then:
```bash
study voice --file my_question.wav --no-tts
```
`--file` transcribes the file instead of using the microphone. `--no-tts` prints the answer without speaking it.

---

## 6. Use your own notes

Put your files into folders named by type. The type is worked out from the folder name:

```
my_course/
  notes/      lecture notes, slides exported as PDF, summaries       → kind "notes"
  exams/      past papers / question sheets                          → kind "exam"
  keys/       answer keys, marking schemes, model solutions          → kind "answer_key"
```

```bash
study ingest ~/my_course/
```

Or set the type yourself:
```bash
study ingest lecture5.pdf --kind notes
study ingest final_2024.pdf --kind exam
study ingest final_2024_solutions.pdf --kind answer_key
```

Tips:
- **Supported formats:** `.pdf`, `.md`, `.markdown`, `.txt`.
- **Scanned PDFs (photos of pages) contain no text,** so they are reported as
  `skipped (no extractable text)`. Run them through OCR first, for example with
  [ocrmypdf](https://ocrmypdf.readthedocs.io/).
- **`check_answer` works best when the answer key names the question,** e.g. "Q2 answer key: …",
  and when you mention the question when asking, e.g. "Check my answer to midterm Q2: …".
- **Edited a file?** Run `ingest` again. Only changed files are re-indexed.
- **Start over:** `docker compose down -v` deletes the database, then run `docker compose up -d` and `study init-db`.

---

## 7. Run the tests (no API key needed)

```bash
pytest
```
Expected output:
```
....................s                                     [100%]
20 passed, 1 skipped
```

- **The unit tests use a fake Claude client and a fake search backend.** They check chunking,
  citations, tool schemas, the agent loop, refusal and truncation handling, and that
  `check_answer` refuses to grade without a key.
- **The skipped test is the database integration test.** To run it, point it at a
  **separate, empty** database. It deletes and recreates the tables.

```bash
docker compose exec db createdb -U study study_test
TEST_DATABASE_URL=postgresql://study:study@localhost:5432/study_test pytest
```
(Windows PowerShell: `$env:TEST_DATABASE_URL="postgresql://study:study@localhost:5432/study_test"; pytest`)

The integration test uses a built-in offline embedder, so it needs no model download.

---

## 8. Configuration

Set these in `.env` or as environment variables. Environment variables take priority.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | required for `ask`, `chat`, `voice` |
| `DATABASE_URL` | `postgresql://study:study@localhost:5432/study` | matches `docker-compose.yml` |
| `CLAUDE_MODEL` | `claude-opus-5-5` | the Claude model used by the agent |
| `CLAUDE_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh`, `max`: higher means more thorough, slower and more expensive |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | must produce 384-dimension vectors (see `config.EMBED_DIM`) |
| `WHISPER_MODEL` | `base.en` | `tiny.en` is faster; `small.en` is more accurate |
| `EMBED_BACKEND` | `sentence-transformers` | `hash` is a test-only offline stand-in. **Don't use it for real notes.** |

**Refusal fallback:** if Claude declines a request, the API automatically retries it on a
fallback model in the same call (`fallbacks="default"`). This is set in `study_assistant/tools.py`
(`Toolbox(..., use_fallbacks=False)` turns it off).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `study: command not found` | Activate the virtual environment (Step 2), then run `pip install -e .` again. |
| `connection refused` / `could not connect to server` | The database isn't running. Start Docker Desktop and run `docker compose up -d`. |
| `port is already allocated` on `docker compose up` | Something else is using port 5432, often a local Postgres. Change the port mapping in `docker-compose.yml` to `"5433:5432"` and set `DATABASE_URL=postgresql://study:study@localhost:5433/study`. |
| `Authentication failed: set ANTHROPIC_API_KEY` | `.env` is missing or the key is empty. Make sure you run `study` from the `study-assistant` folder, where `.env` lives. |
| `Could not load embedding model ...` | The first run needs internet to download the model from Hugging Face. Check your connection, VPN or proxy. |
| `relation "chunks" does not exist` | Run `study init-db`. |
| `type "vector" does not exist` / `extension "vector" is not available` | Your Postgres has no pgvector. Use the Docker setup, or install pgvector (below). |
| `skipped (no extractable text)` when ingesting | The PDF is a scan. Run OCR on it first (see section 6). |
| `PortAudio library not found` | Linux: `sudo apt install libportaudio2`. |
| Voice mode: no sound / `eSpeak` error | Linux: `sudo apt install espeak-ng`. Or use `--no-tts`. |
| Voice transcript is wrong | Speak closer to the mic, or set `WHISPER_MODEL=small.en`. |
| `Rate limited by the API` | Wait the number of seconds shown and try again. |
| Answers don't cite the right page | Run `study search "<your question>"` to see what retrieval returns. If the right page isn't there, rephrase or check that the file was ingested (`study stats`). |

### Postgres without Docker

Install PostgreSQL 14+ and the pgvector extension:
- **macOS:** `brew install postgresql@16 pgvector`
- **Ubuntu/Debian:** `sudo apt install postgresql postgresql-16-pgvector` (match your Postgres version)
- **Windows:** see [pgvector's Windows instructions](https://github.com/pgvector/pgvector#windows)

Then create a user and database, and point `DATABASE_URL` at them:
```bash
sudo -u postgres psql -c "CREATE USER study WITH PASSWORD 'study' SUPERUSER;"
sudo -u postgres createdb -O study study
```
(`SUPERUSER` is needed so `study init-db` can run `CREATE EXTENSION vector`. Alternatively, run
`CREATE EXTENSION vector;` yourself as the postgres user and drop `SUPERUSER`.)

---

## 10. Project structure

```
study-assistant/
├── study_assistant/
│   ├── cli.py          the `study` command (init-db, ingest, stats, search, ask, chat, voice)
│   ├── config.py       settings from .env / environment
│   ├── db.py           PostgreSQL connection + schema (pgvector, HNSW index)
│   ├── embeddings.py   local sentence-transformers embedder (+ offline test stub)
│   ├── ingest.py       load PDF/MD/TXT per page, chunk, embed, store; skip unchanged files
│   ├── retrieval.py    cosine-similarity search, "[file p.N]" citation formatting
│   ├── tools.py        the three tools + structured-output calls for solving and grading
│   ├── agent.py        the tool-calling loop and conversation history
│   └── voice.py        microphone → Whisper → agent → text-to-speech
├── tests/              unit tests (fake Claude client) + pgvector integration test
├── sample_data/        example notes, exam and answer key
├── docker-compose.yml  PostgreSQL 16 + pgvector
├── .env.example        copy to .env and add your API key
└── pyproject.toml      dependencies and the `study` command
```
