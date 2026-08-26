"""
Whisper speech-to-text: API + local faster-whisper demo.
Creates a synthetic WAV file for testing without needing a real audio file.
Requires: openai (for API mode) or faster-whisper (for local mode)
"""
import os
import io
import struct
import math

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("OPENAI_API_KEY not set. Showing Whisper API mechanics.\n")
    LIVE = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    LIVE = True

try:
    from faster_whisper import WhisperModel
    LOCAL_WHISPER = True
except ImportError:
    LOCAL_WHISPER = False


def make_sine_wav(frequency: float = 440.0, duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a WAV file with a sine wave tone (for testing)."""
    n_samples = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
               for i in range(n_samples)]

    buf = io.BytesIO()
    # WAV header
    data_size = n_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))           # chunk size
    buf.write(struct.pack("<H", 1))            # PCM
    buf.write(struct.pack("<H", 1))            # mono
    buf.write(struct.pack("<I", sample_rate))  # sample rate
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))            # block align
    buf.write(struct.pack("<H", 16))           # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for s in samples:
        buf.write(struct.pack("<h", s))
    return buf.getvalue()


if __name__ == "__main__":
    print("=== WHISPER SPEECH-TO-TEXT DEMO ===\n")

    if not LIVE and not LOCAL_WHISPER:
        print("API shapes:\n")
        print("""
from openai import OpenAI
client = OpenAI()

# 1. Basic transcription
with open("meeting.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",          # optional: force language
        prompt="Technical meeting about Kubernetes"  # optional: context hint
    )
print(transcript.text)

# 2. Verbose JSON with word timestamps
with open("interview.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )
for segment in transcript.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# 3. Translation to English (from any language)
with open("french_audio.mp3", "rb") as f:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=f,
    )
print(translation.text)
""")
        print("Pricing: $0.006 per minute ($0.36/hour)")
        print("Context size: up to 25MB per file (use chunking for larger files)")
    elif LOCAL_WHISPER:
        import tempfile

        print("Using local faster-whisper (no API key needed)\n")
        # We can't transcribe a pure sine wave — create a tiny silent WAV
        # In a real test you'd use an actual speech file
        wav_data = make_sine_wav(220.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name

        print(f"Test audio: {tmp_path} ({len(wav_data)} bytes)")
        print("(This is a sine tone — real speech would give meaningful transcription)\n")

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, beam_size=1)

        print(f"Detected language: {info.language} (confidence: {info.language_probability:.1%})")
        for seg in segments:
            print(f"  [{seg.start:.2f}s–{seg.end:.2f}s] {seg.text!r}")

        os.unlink(tmp_path)
        print("\nTo transcribe real audio: pass a .mp3/.wav file path to model.transcribe()")
    elif LIVE:
        print("Using OpenAI Whisper API\n")
        wav_data = make_sine_wav(440.0, 2.0)
        print(f"Test audio: {len(wav_data)} bytes WAV (440Hz tone)\n")
        print("Note: Whisper will likely transcribe silence as empty or gibberish.")
        print("For a real demo, replace wav_data with an actual speech recording.\n")

        # Transcription
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("test.wav", io.BytesIO(wav_data), "audio/wav"),
            )
            print(f"Transcription: {transcript.text!r}")
        except Exception as e:
            print(f"Error: {e}")

        # Verbose JSON with timestamps
        try:
            transcript_v = client.audio.transcriptions.create(
                model="whisper-1",
                file=("test.wav", io.BytesIO(wav_data), "audio/wav"),
                response_format="verbose_json",
            )
            print(f"Language: {transcript_v.language}")
            print(f"Duration: {transcript_v.duration:.1f}s")
            for seg in (transcript_v.segments or []):
                print(f"  [{seg['start']:.1f}–{seg['end']:.1f}s] {seg['text']!r}")
        except Exception as e:
            print(f"Verbose JSON error: {e}")
