"""
Table extraction from PDFs: detect, parse, export to CSV, and query with LLM.
Requires: pdfplumber  (pip install pdfplumber)
Optional: reportlab for sample PDF generation
"""
import io
import os
import csv

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")


# ─── Sample PDF with tables ───────────────────────────────────────────────────

def make_pdf_with_tables() -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Q1 2024 Performance Report", styles["Heading1"]))
    story.append(Paragraph("Revenue by Product Line", styles["Heading2"]))

    revenue_data = [
        ["Product", "Q1 Revenue", "Q4 Revenue", "Growth %"],
        ["API Platform", "$1,240,000", "$980,000", "+26.5%"],
        ["Enterprise", "$890,000", "$750,000", "+18.7%"],
        ["Starter Plans", "$320,000", "$290,000", "+10.3%"],
        ["Professional", "$540,000", "$480,000", "+12.5%"],
        ["TOTAL", "$2,990,000", "$2,500,000", "+19.6%"],
    ]

    t1 = Table(revenue_data, colWidths=[120, 100, 100, 80])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, -1), (-1, -1), colors.lightgrey),
    ]))
    story.append(t1)

    story.append(Paragraph(" ", styles["Normal"]))
    story.append(Paragraph("Team Headcount by Department", styles["Heading2"]))

    headcount_data = [
        ["Department", "Jan", "Feb", "Mar", "Delta"],
        ["Engineering", "42", "45", "48", "+6"],
        ["Sales", "18", "20", "22", "+4"],
        ["Marketing", "8", "8", "10", "+2"],
        ["Support", "12", "13", "13", "+1"],
        ["Total", "80", "86", "93", "+13"],
    ]

    t2 = Table(headcount_data, colWidths=[120, 60, 60, 60, 60])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(t2)

    doc.build(story)
    return buf.getvalue()


# ─── Table extraction ─────────────────────────────────────────────────────────

def extract_all_tables(pdf_bytes: bytes) -> list[dict]:
    results = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for tbl_idx, table in enumerate(tables):
                # Clean: strip whitespace, replace None with ""
                cleaned = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                ]
                results.append({
                    "page": page_num,
                    "table_index": tbl_idx,
                    "rows": len(cleaned),
                    "cols": len(cleaned[0]) if cleaned else 0,
                    "data": cleaned,
                })
    return results


def table_to_csv(table_data: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(table_data)
    return buf.getvalue()


def table_to_markdown(table_data: list[list[str]]) -> str:
    if not table_data:
        return ""
    header = table_data[0]
    separator = ["---"] * len(header)
    rows = table_data[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def query_table_with_llm(table_md: str, question: str) -> str:
    """Ask an LLM a question about a table (tries OpenAI, Anthropic, Ollama)."""
    prompt = f"""Here is a data table in Markdown format:

{table_md}

Question: {question}

Answer based only on the data in the table. Be concise."""

    if OPENAI_KEY:
        try:
            import httpx
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                },
                timeout=30,
            )
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"OpenAI error: {e}"

    if ANTHROPIC_KEY:
        try:
            import httpx
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            return r.json()["content"][0]["text"].strip()
        except Exception as e:
            return f"Anthropic error: {e}"

    try:
        import httpx
        r = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=60,
        )
        return r.json()["response"].strip()
    except Exception as e:
        return f"Ollama error: {e}"


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== TABLE EXTRACTION DEMO ===\n")

if not PDFPLUMBER_AVAILABLE:
    print("pdfplumber not installed. pip install pdfplumber\n")
    print("""
Table extraction API:

import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        # tables = [ [[row1col1, row1col2, ...], [row2col1, ...]], ... ]

        for table in tables:
            header = table[0]   # first row = header
            for row in table[1:]:
                record = dict(zip(header, row))
                print(record)

# Fine-tune detection with table settings:
settings = {
    "vertical_strategy": "lines",     # or "text", "explicit"
    "horizontal_strategy": "lines",   # or "text", "explicit"
    "snap_tolerance": 3,               # pixels
    "join_tolerance": 3,
}
table = page.extract_table(table_settings=settings)
""")
    raise SystemExit(0)

PDF_PATH = os.environ.get("PDF_PATH", "")

if PDF_PATH and os.path.exists(PDF_PATH):
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    print(f"Loaded: {PDF_PATH}")
elif REPORTLAB_AVAILABLE:
    pdf_bytes = make_pdf_with_tables()
    print(f"Generated sample PDF with tables ({len(pdf_bytes):,} bytes)")
else:
    print("No PDF found. Set PDF_PATH, or: pip install reportlab")
    raise SystemExit(0)

tables = extract_all_tables(pdf_bytes)
print(f"Found {len(tables)} table(s) across {len(set(t['page'] for t in tables))} page(s)\n")

for t in tables:
    print(f"Table {t['table_index']+1} on page {t['page']}: {t['rows']} rows × {t['cols']} cols")
    md = table_to_markdown(t["data"])
    print(md)

    # Export to CSV
    csv_str = table_to_csv(t["data"])
    csv_lines = csv_str.strip().split("\n")
    print(f"\nCSV ({len(csv_lines)} rows):")
    for line in csv_lines[:3]:
        print(f"  {line}")
    if len(csv_lines) > 3:
        print(f"  ... ({len(csv_lines)-3} more rows)")
    print()

# LLM Q&A on first table
if tables:
    first_md = table_to_markdown(tables[0]["data"])
    questions = [
        "Which product had the highest revenue in Q1?",
        "What was the total revenue growth percentage?",
    ]
    print("─── LLM Table Q&A ───")
    for q in questions:
        print(f"Q: {q}")
        answer = query_table_with_llm(first_md, q)
        print(f"A: {answer}\n")
