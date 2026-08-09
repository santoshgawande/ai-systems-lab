# Lab 02 — OpenAI Text-to-Speech

Convert text to natural-sounding audio with multiple voices and streaming output.

## What you learn

- OpenAI TTS API: 6 voices, 3 models, streaming support
- Write audio to file (MP3, WAV, OPUS, FLAC, PCM)
- Stream audio chunks as they arrive (for low-latency playback)
- Build a voice pipeline: text → TTS → play on Mac

## Run

```bash
export OPENAI_API_KEY=sk-...
python tts.py
```

## API shape

```python
from openai import OpenAI
client = OpenAI()

# Basic TTS — save to file
response = client.audio.speech.create(
    model="tts-1",          # tts-1 (fast) or tts-1-hd (higher quality)
    voice="nova",           # alloy, echo, fable, onyx, nova, shimmer
    input="Hello world! This is a text to speech test.",
    speed=1.0,              # 0.25–4.0
    response_format="mp3",  # mp3, opus, aac, flac, wav, pcm
)
response.stream_to_file("output.mp3")

# Streaming (low-latency: first audio chunk arrives quickly)
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input=long_text,
) as response:
    response.stream_to_file("output.mp3")

# Play on macOS (no extra libraries)
import subprocess
subprocess.run(["afplay", "output.mp3"])
```

## Voice guide

| Voice | Character | Best for |
|---|---|---|
| alloy | Neutral, balanced | Default, general purpose |
| echo | Warm, male | Storytelling, podcasts |
| fable | British-ish, expressive | Long-form, audiobooks |
| onyx | Deep, authoritative | Corporate, formal |
| nova | Clear, female, natural | Customer support, education |
| shimmer | Soft, female | Gentle content, meditation |

## Pricing

- tts-1: $15 per 1M characters
- tts-1-hd: $30 per 1M characters
- 1000 words ≈ 5000 chars ≈ $0.075 (tts-1)
