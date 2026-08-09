"""
OpenAI TTS: generate speech from text, compare voices, stream audio.
Saves to /tmp/ and optionally plays via macOS afplay.
Requires: OPENAI_API_KEY
"""
import os
import sys
import subprocess
import tempfile

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing TTS API mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

PLAY_AUDIO = sys.platform == "darwin" and os.environ.get("PLAY_AUDIO", "0") == "1"

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

VOICE_DESCRIPTIONS = {
    "alloy":   "Neutral, balanced — good all-purpose voice",
    "echo":    "Warm, male — good for storytelling",
    "fable":   "Expressive — good for long-form content",
    "onyx":    "Deep, authoritative — good for formal content",
    "nova":    "Natural, female — good for education/support",
    "shimmer": "Soft, gentle — good for calm content",
}

# ─── Sample texts ─────────────────────────────────────────────────────────────

TEXTS = {
    "short": "Hello! This is a text-to-speech demo using the OpenAI TTS API.",
    "technical": (
        "The Transformer architecture uses multi-head self-attention mechanisms "
        "to process sequences in parallel, unlike recurrent neural networks which "
        "process tokens sequentially. This parallelism enables faster training on GPUs."
    ),
    "assistant": (
        "I've completed the analysis. I found three main issues in your code: "
        "a potential SQL injection vulnerability on line 47, "
        "a missing null check in the user authentication flow, "
        "and an N+1 query problem in the products endpoint. "
        "I recommend addressing the security issues first."
    ),
}


def generate_speech(text: str, voice: str, model: str = "tts-1") -> bytes:
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return response.content


def save_and_play(audio_bytes: bytes, filename: str) -> str:
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    if PLAY_AUDIO:
        subprocess.run(["afplay", path], check=True)
    return path


def estimate_cost(text: str, model: str) -> float:
    per_million = 15.0 if model == "tts-1" else 30.0
    return len(text) / 1_000_000 * per_million


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== OPENAI TTS DEMO ===\n")

if not LIVE:
    print("TTS API shapes:\n")
    print("""
from openai import OpenAI
client = OpenAI()

# 1. Basic — save to file
response = client.audio.speech.create(
    model="tts-1",       # or "tts-1-hd" (higher quality, 2x cost)
    voice="nova",        # alloy, echo, fable, onyx, nova, shimmer
    input="Hello world",
    speed=1.0,           # 0.25–4.0
)
response.stream_to_file("hello.mp3")

# 2. Get bytes directly
audio_bytes = response.content
with open("hello.mp3", "wb") as f:
    f.write(audio_bytes)

# 3. Streaming (lower latency for long texts — first chunk arrives faster)
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input=long_text,
    response_format="mp3",
) as response:
    with open("output.mp3", "wb") as f:
        for chunk in response.iter_bytes(1024):
            f.write(chunk)  # or pipe to audio player

# 4. Play on macOS
import subprocess
subprocess.run(["afplay", "output.mp3"])

# 5. Play on Linux
subprocess.run(["mpg123", "output.mp3"])  # or aplay for WAV
""")
    print("Voice comparison:")
    for voice in VOICES:
        print(f"  {voice:<10} {VOICE_DESCRIPTIONS[voice]}")
    print()
    print("Models:")
    print("  tts-1:    $0.015/1K chars — fast, good quality")
    print("  tts-1-hd: $0.030/1K chars — higher quality, slower")
    print()
    print("Set PLAY_AUDIO=1 to auto-play on macOS: PLAY_AUDIO=1 python tts.py")
else:
    # Demo 1: All 6 voices on the same text
    print("─── Voice comparison (all 6 voices) ───\n")
    sample_text = TEXTS["short"]
    print(f"Text: {sample_text!r}\n")

    for voice in VOICES:
        audio = generate_speech(sample_text, voice)
        path = save_and_play(audio, f"tts-{voice}.mp3")
        cost = estimate_cost(sample_text, "tts-1")
        print(f"  {voice:<10} {len(audio):>7} bytes  ${cost:.6f}  → {path}")
        if PLAY_AUDIO:
            import time; time.sleep(3)

    print()

    # Demo 2: tts-1 vs tts-1-hd quality comparison
    print("─── Model comparison: tts-1 vs tts-1-hd ───\n")
    technical_text = TEXTS["technical"]
    print(f"Text ({len(technical_text)} chars): {technical_text[:60]}...\n")

    for model in ["tts-1", "tts-1-hd"]:
        audio = generate_speech(technical_text, "nova", model=model)
        path = save_and_play(audio, f"{model}-technical.mp3")
        cost = estimate_cost(technical_text, model)
        print(f"  {model:<10} {len(audio):>8} bytes  ${cost:.6f}  → {path}")
    print()

    # Demo 3: Assistant-style TTS with streaming (save to file)
    print("─── Streaming TTS (low-latency) ───\n")
    assistant_text = TEXTS["assistant"]
    print(f"Text ({len(assistant_text)} chars): {assistant_text[:80]}...\n")

    path = os.path.join(tempfile.gettempdir(), "tts-stream.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=assistant_text,
        response_format="mp3",
    ) as response:
        chunk_count = 0
        with open(path, "wb") as f:
            for chunk in response.iter_bytes(4096):
                f.write(chunk)
                chunk_count += 1
    print(f"  Streamed {chunk_count} chunks → {path}")
    print(f"  Cost: ${estimate_cost(assistant_text, 'tts-1'):.6f}")

    if PLAY_AUDIO:
        subprocess.run(["afplay", path])

    print()
    print("Tip: enable audio playback → PLAY_AUDIO=1 python tts.py")
