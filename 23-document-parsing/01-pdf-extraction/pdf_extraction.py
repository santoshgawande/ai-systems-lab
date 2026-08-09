"""
PDF text extraction for RAG pipelines.
Generates a sample PDF, extracts it page-by-page, chunks with overlap.
Requires: pdfplumber  (pip install pdfplumber)
"""
import io
import os
import re

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Try to generate a sample PDF using reportlab (optional)
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─── Sample PDF generator ─────────────────────────────────────────────────────

SAMPLE_CONTENT = [
    ("Introduction to Python Error Handling", """
Python provides structured exception handling through try/except blocks.
Exceptions are objects that represent error conditions. When an exception
is raised, Python unwinds the call stack looking for a matching handler.

The base class for all exceptions is BaseException. Most user exceptions
should inherit from Exception. Built-in exceptions include ValueError,
TypeError, FileNotFoundError, and ConnectionError.
"""),
    ("Database Connection Management", """
Database connections are expensive resources. Always use context managers
or connection pools to manage them. The psycopg2 library provides
OperationalError for connection failures and IntegrityError for constraint
violations.

Connection pooling with SQLAlchemy prevents connection exhaustion under
high load. Configure pool_size=5 and max_overflow=10 for most applications.
Pool pre_ping=True checks connections are alive before use.
"""),
    ("Logging Best Practices", """
Structured logging with consistent fields enables efficient log aggregation.
Include request_id, user_id, and service_name in every log entry.

Use log levels appropriately: DEBUG for development traces, INFO for normal
operations, WARNING for unexpected but handled situations, ERROR for failures
that affect users, CRITICAL for system-threatening conditions.

Never log sensitive data: passwords, API keys, PII. Use log sampling for
high-throughput paths to avoid I/O bottlenecks.
"""),
]


def make_sample_pdf() -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    for title, body in SAMPLE_CONTENT:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 60, title)
        c.setFont("Helvetica", 11)
        y = height - 90
        for line in body.strip().split("\n"):
            line = line.strip()
            if not line:
                y -= 8
                continue
            # Simple word wrap at ~80 chars
            words = line.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > 80:
                    c.drawString(50, y, current)
                    y -= 16
                    current = word
                else:
                    current = f"{current} {word}".strip()
            if current:
                c.drawString(50, y, current)
                y -= 16
        c.showPage()

    c.save()
    return buf.getvalue()


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page": i + 1,
                "text": text,
                "chars": len(text),
                "bbox": page.bbox,
            })
    return pages


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split text into overlapping chunks for RAG indexing."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        # Try to end on sentence boundary
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            end = start + last_period + 1
            chunk = text[start:end]
        chunks.append({"chunk_id": idx, "text": chunk, "start": start, "end": end})
        idx += 1
        start = end - overlap
    return chunks


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== PDF EXTRACTION DEMO ===\n")

if not PDFPLUMBER_AVAILABLE:
    print("pdfplumber not installed. pip install pdfplumber\n")
    print("""
Key API:

import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    print(f"Pages: {len(pdf.pages)}")

    for page in pdf.pages:
        text = page.extract_text()          # plain text
        tables = page.extract_tables()      # list of table data
        words = page.extract_words()        # with bounding boxes

        # Page metadata
        print(page.width, page.height)      # dimensions in points
        print(page.bbox)                    # (x0, y0, x1, y1)

# Crop to a region (useful for headers/footers removal)
cropped = page.crop((0, 50, page.width, page.height - 50))
text = cropped.extract_text()
""")
    raise SystemExit(0)

# Generate or load PDF
PDF_PATH = os.environ.get("PDF_PATH", "")

if PDF_PATH and os.path.exists(PDF_PATH):
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    print(f"Loaded: {PDF_PATH} ({len(pdf_bytes):,} bytes)")
elif REPORTLAB_AVAILABLE:
    pdf_bytes = make_sample_pdf()
    print(f"Generated sample PDF ({len(pdf_bytes):,} bytes, {len(SAMPLE_CONTENT)} pages)")
else:
    print("No PDF found. Set PDF_PATH env var, or: pip install reportlab")
    print("Continuing with API demo only...")
    pdf_bytes = b""

if pdf_bytes:
    # Extract pages
    pages = extract_pages(pdf_bytes)
    print(f"\nExtracted {len(pages)} pages:\n")
    for p in pages:
        print(f"  Page {p['page']}: {p['chars']} chars")
        preview = p["text"][:120].replace("\n", " ")
        print(f"    Preview: {preview}...")

    # Chunk for RAG
    all_text = " ".join(p["text"] for p in pages)
    chunks = chunk_text(all_text, chunk_size=400, overlap=80)
    print(f"\nChunking (size=400, overlap=80): {len(chunks)} chunks\n")
    for c in chunks[:3]:
        print(f"  Chunk {c['chunk_id']}: chars {c['start']}-{c['end']}")
        print(f"    {c['text'][:100]}...")
        print()

    print("RAG pipeline pattern:")
    print("  1. extract_pages() → page text with metadata")
    print("  2. chunk_text()    → overlapping chunks")
    print("  3. embed each chunk (Ollama / OpenAI)")
    print("  4. store in vector DB (Qdrant / pgvector)")
    print("  5. query → top-k chunks → LLM context")
