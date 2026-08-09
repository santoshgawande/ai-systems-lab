# Section 23 — Document Parsing

Extract text, tables, and structured data from PDFs and other documents for RAG pipelines.

## What you learn

- PDF text extraction: pdfplumber vs PyMuPDF (fitz) — speed vs accuracy trade-offs
- Table detection and extraction from PDFs
- Feeding extracted text into RAG pipelines

## Labs

| Lab | What it covers |
|---|---|
| 01-pdf-extraction | pdfplumber text extraction, page-by-page chunking, metadata |
| 02-table-extraction | Table detection, CSV export, LLM table Q&A |

## Setup

```bash
pip install -r requirements.txt
```

## When you need document parsing

- RAG over internal knowledge bases (PDFs, Word docs)
- Financial report ingestion (tables, numbers)
- Contract analysis (structured clause extraction)
- Scientific paper pipelines (abstract, methods, results sections)
