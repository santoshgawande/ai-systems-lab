# Lab 02 — Table Extraction

Extract tables from PDFs, export to CSV/Markdown, and query with an LLM.

## What you learn

- `pdfplumber.extract_tables()` — automatic table detection
- Cleaning and normalising extracted table data
- Exporting to CSV and Markdown formats
- LLM table Q&A: pass Markdown table as context, ask questions

## Run

```bash
pip install pdfplumber
pip install reportlab   # optional: generates sample PDF with tables
python table_extraction.py

# Your own PDF:
PDF_PATH=/path/to/report.pdf python table_extraction.py
```

## Key API

```python
import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        # [[row1col1, row1col2], [row2col1, row2col2], ...]

        for table in tables:
            header = table[0]
            for row in table[1:]:
                record = dict(zip(header, row))

# Fine-tune detection (for tricky layouts)
table = page.extract_table(table_settings={
    "vertical_strategy": "lines",    # or "text", "explicit"
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
})
```

## LLM table Q&A pattern

```
1. extract_tables() → raw data
2. table_to_markdown() → Markdown string
3. LLM prompt: "Here is a table:\n{md}\n\nQ: {question}"
4. LLM answers based on structured data
```

This works reliably because LLMs understand Markdown tables natively.
