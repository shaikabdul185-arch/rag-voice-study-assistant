# RAG + Voice Study Assistant

A retrieval-augmented, tool-calling study agent over your own CS notes and past exam papers.

- **RAG pipeline** – PDFs / Markdown / text are split into page-aware chunks, embedded locally
  with sentence-transformers, and stored in PostgreSQL with **pgvector**. Every answer cites
  the source page, e.g. `[operating_systems.pdf p.2]`.
- **Agent loop with tool calling** – Claude chooses between three tools:
  `search_notes`, `solve_step_by_step`, and `check_answer` (grades a student's answer
  against the answer key, and refuses to grade when no key matches).
- **Voice mode** – speech-to-text with Whisper (`faster-whisper`, local) in, text-to-speech
  (`pyttsx3`, offline) out.

**Stack:** Python · Claude API (tool use, structured outputs) · PostgreSQL + pgvector ·
sentence-transformers · faster-whisper · pyttsx3

## Architecture

```
            ┌──────────── ingest ────────────┐
 PDF/MD ──► │ per-page text → chunks (≤1000c, │──► PostgreSQL + pgvector
            │ 150c overlap) → MiniLM (384-d)  │    documents(kind: notes|exam|answer_key)
            └─────────────────────────────────┘    chunks(page, content, embedding)
                                                          ▲
 mic ─► Whisper STT ─┐                                    │ cosine search (HNSW)
                     ▼                                    │
 CLI question ──► StudyAgent loop (Claude) ──tool_use──► search_notes ────────┘
                     ▲     │                         ├─► solve_step_by_step ─► Claude (structured steps)
                     │     └──tool_result────────────┴─► check_answer ──────► Claude grader vs. key
                     ▼
            cited answer ──► (voice) citations → "from X, page N" ──► TTS
```

| File | Role |
|---|---|
| `study_assistant/ingest.py` | load pages, chunk without crossing pages, hash-based re-ingest |
| `study_assistant/retrieval.py` | pgvector cosine search, `[file p.N]` citation formatting |
| `study_assistant/tools.py` | the three `@beta_tool` tools + structured-output sub-calls |
| `study_assistant/agent.py` | the tool-calling loop, multi-turn history, refusal/truncation handling |
| `study_assistant/voice.py` | record → transcribe → ask → speak |
| `study_assistant/cli.py` | `study` command |

## Setup

```bash
cd study-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[voice,dev]"          # drop ",voice" if you don't need voice mode

docker compose up -d                   # Postgres 16 + pgvector on localhost:5432
cp .env.example .env                   # then set ANTHROPIC_API_KEY
study init-db
```

Voice mode also needs system audio libraries: PortAudio (`sudo apt install libportaudio2`)
for the microphone and eSpeak (`sudo apt install espeak-ng`) for pyttsx3 on Linux.
macOS and Windows use their built-in speech engines.

## Usage

```bash
# Index your material. The collection is inferred from folder names (notes/, exams/, keys/)
# or set explicitly with --kind notes|exam|answer_key.
study ingest sample_data/
study ingest ~/uni/cs-notes --kind notes
study stats

# Debug retrieval without calling the LLM
study search "conditions for deadlock" -k 3

# Ask questions
study ask "What is the height of a B-tree and why do databases use them?"
study ask "Solve: FCFS and SJF average waiting time for bursts 24, 3, 3"
study ask "Check my answer to midterm Q1: mutual exclusion, hold and wait, no preemption"

study chat                             # multi-turn; /reset, /quit
study voice                            # press Enter to talk, Enter to stop
study voice --file question.wav --no-tts
```

Re-running `ingest` skips unchanged files (SHA-256) and re-indexes changed ones.

## Configuration (env or `.env`)

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | or an `ant auth login` profile |
| `DATABASE_URL` | `postgresql://study:study@localhost:5432/study` | |
| `CLAUDE_MODEL` | `claude-opus-5-5` | |
| `CLAUDE_EFFORT` | `medium` | `low` … `max`; trades depth for cost/latency |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | must output 384-d vectors (see `config.EMBED_DIM`) |
| `WHISPER_MODEL` | `base.en` | e.g. `small.en` for better accuracy |

**Refusal fallback:** requests opt in to the API's server-side fallback
(`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). If the model declines a
request, the API retries it on a fallback model in the same call. Turn it off with
`Toolbox(..., use_fallbacks=False)`.

## Tests

```bash
pytest                                              # unit tests, no network or DB needed
TEST_DATABASE_URL=postgresql://... pytest           # + pgvector integration test (drops tables! use a scratch DB)
```

The unit tests use a fake Anthropic client and a fake retriever. The integration test uses an
offline hash embedder (`EMBED_BACKEND=hash`, for tests only), so no model download is needed.
