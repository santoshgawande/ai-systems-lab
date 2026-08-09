# Lab 01 — Whisper Speech-to-Text

Transcribe audio to text using OpenAI's Whisper API or local faster-whisper.

## What you learn

- Whisper API: transcribe audio files, get word-level timestamps
- faster-whisper: local inference, no API key, runs on Mac Studio
- Supported formats: mp3, mp4, wav, webm, ogg, flac (max 25MB for API)
- Language detection and translation to English
- When to use API vs local

## Run

```bash
export OPENAI_API_KEY=sk-...
python whisper_stt.py

# For local whisper:
pip install faster-whisper
python whisper_stt.py --local
```

## API shape

```python
from openai import OpenAI
client = OpenAI()

# Basic transcription
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
    )
print(transcript.text)

# With timestamps (word-level)
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
for word in transcript.words:
    print(f"  {word.start:.2f}s–{word.end:.2f}s: {word.word}")

# Translation to English (from any language)
with open("french_audio.mp3", "rb") as f:
    result = client.audio.translations.create(
        model="whisper-1",
        file=f,
    )
print(result.text)  # English translation
```

## Local faster-whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.mp3")
for segment in segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

## Pricing (API)

- whisper-1: $0.006 per minute
- 1 hour of audio = $0.36
- Local: free after hardware
