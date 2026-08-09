# 06 — Vision Models

Image understanding with local vision models via Ollama.
No cloud API needed — all runs on Mac Studio.

## Setup

Pull vision models first:
```bash
ollama pull llava              # 4.7GB — classic, solid all-rounder
ollama pull moondream          # 1.8GB — tiny, fast, good for quick tasks
ollama pull qwen2.5vl:7b       # 6.0GB — strong at OCR + structured output
```

Install extra dependency (image resizing):
```bash
# from 01-llm-apis/
pip install -r requirements.txt
```

Add test images to `samples/` before running (any JPG/PNG works).
See `samples/README.md` for what kinds of images to try.

## Labs

| Script | Run | What you learn |
|---|---|---|
| `image_qa.py` | `python image_qa.py samples/photo.jpg "What is in this image?"` | Raw vision API call, image encoding |
| `compare_models.py` | `python compare_models.py samples/photo.jpg "Describe this"` | Compare llava vs moondream vs qwen on same image |
| `ocr_extract.py` | `python ocr_extract.py samples/screenshot.png` | Extract text and structured data from images |
| `batch_describe.py` | `python batch_describe.py samples/ "What objects are visible?"` | Process a folder of images, save results |

## Supported Models

| Model | Ollama tag | Good at |
|---|---|---|
| LLaVA | `llava` | General image Q&A, scene description |
| Moondream | `moondream` | Fast captions, simple questions |
| Qwen2.5-VL | `qwen2.5vl:7b` | OCR, charts, structured extraction |
| LLaVA 13B | `llava:13b` | Higher quality than 7B |

## How vision models work (what you're learning)

Images are base64-encoded and sent inside the `messages` array alongside the text prompt.
The model has a vision encoder (CLIP or similar) that converts the image into token embeddings,
then the language model sees both image tokens and text tokens together.

This is the same mechanism used by GPT-4o vision, Claude vision, and Gemini multimodal.
