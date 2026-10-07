# rag-voice-study-assistant

A retrieval-augmented, tool-calling study assistant: ask questions about your own CS notes and
past exam papers and get answers that cite the source page, worked step-by-step solutions, and
answers graded against the answer key, by text or by voice.

**Stack:** Python · Claude API (tool use) · PostgreSQL + pgvector · sentence-transformers ·
Whisper speech-to-text · text-to-speech

## Projects

- [`study-assistant/`](study-assistant/): the RAG + voice study assistant. See its
  [README](study-assistant/README.md) for setup, usage, and how to run it with your own API key.

## Quick start

```bash
git clone https://github.com/shaikabdul185-arch/rag-voice-study-assistant.git
cd rag-voice-study-assistant/study-assistant
```

Then follow [study-assistant/README.md](study-assistant/README.md#3-setup-step-by-step).

## License

Released under the [MIT License](LICENSE).
