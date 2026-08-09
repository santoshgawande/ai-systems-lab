# Section 20 — Audio: Whisper STT + TTS

The full speech pipeline: audio → text (Whisper) and text → speech (TTS).

## What you learn

- OpenAI Whisper — speech-to-text, local vs API, word-level timestamps
- OpenAI TTS — text-to-speech, voice options, streaming audio
- Building a voice pipeline: microphone → Whisper → LLM → TTS → speaker
- Faster-Whisper (local) for offline, low-latency transcription

## Labs

| Lab | What it covers |
|---|---|
| 01-whisper | Whisper STT: transcribe audio via API + local faster-whisper |
| 02-openai-tts | TTS: generate speech, stream audio, voice comparison |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# For local Whisper: pip install faster-whisper
```

## Whisper model tiers

| Model | Size | Speed | Accuracy | Use when |
|---|---|---|---|---|
| whisper-1 (API) | Cloud | Fast | High | Production, no local infra |
| faster-whisper tiny | ~75MB | Very fast | Low | Real-time transcription |
| faster-whisper base | ~145MB | Fast | Medium | Good latency/quality balance |
| faster-whisper large-v3 | ~3GB | Slow | SOTA | Accuracy-first offline |

## TTS voices (OpenAI)

| Voice | Character | Best for |
|---|---|---|
| alloy | Neutral | Default, general |
| echo | Male, warm | Narration |
| fable | Male, authoritative | Long-form content |
| onyx | Deep, authoritative | Official/formal |
| nova | Female, natural | Customer-facing |
| shimmer | Female, clear | Educational |

## Voice pipeline architecture

```
Microphone
  → VAD (voice activity detection)
  → Whisper (STT)
  → LLM (process text)
  → TTS
  → Speaker
```

Latency budget for real-time voice:
- VAD: ~50ms
- Whisper (tiny): ~200ms
- LLM (TTFT): ~300ms
- TTS (streaming): first chunk ~200ms
- Total: ~750ms — acceptable for voice assistant
