# Lab 02 — Gemini Multimodal

Gemini natively processes text, images, audio, and video in a single request — no separate embedding step.

## What you learn

- How to send images alongside text prompts
- How Gemini reasons across image + text together
- Generating images from programmatic inputs (PIL)
- Comparing multimodal vs text-only responses on visual tasks
- File API for larger media (images > 20MB, video, audio)

## Run

```bash
export GEMINI_API_KEY=...
python multimodal.py
```

## API shapes

### Image from URL (via httpx fetch)

```python
import httpx, PIL.Image, io

image_data = httpx.get(image_url).content
image = PIL.Image.open(io.BytesIO(image_data))

response = model.generate_content([
    "Describe what you see in this image.",
    image   # PIL.Image or genai.types.Part
])
```

### Image from bytes

```python
import google.generativeai as genai

response = model.generate_content([
    "What text is in this screenshot?",
    {"mime_type": "image/png", "data": image_bytes}
])
```

### Multiple images

```python
response = model.generate_content([
    "Compare these two diagrams:",
    image1,
    image2,
    "Which one is clearer and why?"
])
```

## Supported input types

| Type | Max size (inline) | Notes |
|---|---|---|
| Image (JPEG/PNG/WEBP) | 20 MB | Use File API above |
| PDF | 50 MB | Each page = ~258 tokens |
| Audio (MP3/WAV/etc) | 20 MB | 1 sec ≈ 25 tokens |
| Video (MP4/etc) | 1 GB (File API) | 1 sec ≈ 263 tokens |
