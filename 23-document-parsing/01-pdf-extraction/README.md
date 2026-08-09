# Lab 01 — PDF Text Extraction

Extract and chunk PDF text for RAG ingestion.

## What you learn

- `pdfplumber` page-by-page text extraction with metadata
- Overlapping chunk strategy for RAG (prevents splitting mid-sentence context)
- Cropping pages to remove headers/footers
- Full pipeline: PDF → pages → chunks → embed → vector DB

## Run

```bash
pip install pdfplumber
pip install reportlab   # optional: to generate sample PDF
python pdf_extraction.py

# Use your own PDF:
PDF_PATH=/path/to/document.pdf python pdf_extraction.py
```

## Key API

```python
import pdfplumber

with pdfplumber.open("doc.pdf") as pdf:
    print(len(pdf.pages))

    for page in pdf.pages:
        text = page.extract_text()          # full page text
        tables = page.extract_tables()      # table data as list-of-lists
        words = page.extract_words()        # [{text, x0, y0, x1, y1}, ...]

        # Remove headers/footers (crop 50pt from top and bottom)
        body = page.crop((0, 50, page.width, page.height - 50))
        clean_text = body.extract_text()
```

## pdfplumber vs alternatives

| Library | Best for | Speed |
|---------|----------|-------|
| pdfplumber | Tables, precise extraction | Medium |
| PyMuPDF (fitz) | Speed, scanned PDFs (OCR) | Fast |
| pypdf | Simple text-only, no deps | Fast |
| Docling | Complex layouts, multi-format | Slow |
