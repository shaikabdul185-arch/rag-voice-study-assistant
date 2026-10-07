"""Voice mode: speech-to-text (faster-whisper) -> agent -> text-to-speech (pyttsx3).

All audio libraries are imported lazily so the rest of the package works without the
`[voice]` extra installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000

_CITATION = re.compile(r"\[([^\[\]]+?) p\.(\d+)\]")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def speakable(text: str) -> str:
    """Turn a markdown answer with [file p.N] citations into something pleasant to hear."""
    text = _CODE_FENCE.sub(" (see the code on screen) ", text)
    text = _CITATION.sub(lambda m: f"(from {Path(m.group(1)).stem.replace('_', ' ')}, page {m.group(2)})", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)            # inline code
    text = re.sub(r"[*_#>]+", "", text)                 # emphasis / headings / quotes
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.M)  # bullet markers
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    return re.sub(r"\s+", " ", text).strip()


class Transcriber:
    def __init__(self, model_name: str = "base.en"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio: Any) -> str:
        """`audio` is a path to an audio file or a float32 mono numpy array at 16 kHz."""
        segments, _info = self.model.transcribe(audio, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


def record_until_enter() -> Any:
    """Record from the default microphone; press Enter to stop."""
    import numpy as np
    import sounddevice as sd

    frames: list[Any] = []

    def callback(indata, _frames, _time, status):  # noqa: ANN001
        if status:
            print(f"[audio] {status}")
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("Recording... press Enter to stop. ")
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)[:, 0]


class Speaker:
    def __init__(self, rate: int = 180):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()


def run_voice(agent: Any, whisper_model: str, audio_file: str | None = None, tts: bool = True) -> None:
    from .agent import friendly_api_error

    print("Loading speech-to-text model...")
    stt = Transcriber(whisper_model)
    speaker = Speaker() if tts else None

    def handle(audio: Any) -> None:
        question = stt.transcribe(audio)
        if not question:
            print("(didn't catch that)")
            return
        print(f"\nYou said: {question}")
        try:
            reply = agent.ask(question)
        except Exception as exc:
            print(friendly_api_error(exc))
            return
        print(f"\nAssistant: {reply.text}\n")
        if speaker:
            speaker.say(speakable(reply.text))

    if audio_file:
        handle(audio_file)
        return

    print("Voice mode. Press Enter to start talking, Ctrl+C to quit.")
    try:
        while True:
            input("\nPress Enter to speak... ")
            handle(record_until_enter())
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
