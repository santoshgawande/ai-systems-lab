"""
Gemini multimodal demo: send images + text, reason across modalities.
Creates test images with PIL, then queries Gemini about them.
Requires: GEMINI_API_KEY, Pillow
"""
import os
import io
import math

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("GEMINI_API_KEY not set. Showing multimodal API mechanics.\n")
    LIVE = False
else:
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    LIVE = True

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow not installed. Run: pip install Pillow\n")

MODEL = "gemini-1.5-flash"


# ─── Image generators ─────────────────────────────────────────────────────────

def make_chart_image() -> "Image.Image":
    """Create a simple bar chart image."""
    width, height = 500, 350
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((10, 10), "Monthly Sales Q1 2024", fill="black")

    # Data
    data = [("Jan", 45000), ("Feb", 62000), ("Mar", 58000)]
    bar_width = 80
    max_val = 70000
    bar_area_height = 250
    bar_area_top = 60
    colors = ["#4A90D9", "#E67E22", "#2ECC71"]

    for i, (label, value) in enumerate(data):
        x = 60 + i * (bar_width + 40)
        bar_height = int((value / max_val) * bar_area_height)
        y_top = bar_area_top + bar_area_height - bar_height
        y_bot = bar_area_top + bar_area_height
        draw.rectangle([x, y_top, x + bar_width, y_bot], fill=colors[i])
        draw.text((x + 20, y_bot + 5), label, fill="black")
        draw.text((x + 10, y_top - 18), f"${value//1000}K", fill="black")

    # Y-axis
    draw.line([50, bar_area_top, 50, bar_area_top + bar_area_height], fill="black", width=2)
    draw.line([50, bar_area_top + bar_area_height, 430, bar_area_top + bar_area_height], fill="black", width=2)

    return img


def make_diagram_image() -> "Image.Image":
    """Create a simple architecture diagram."""
    width, height = 600, 400
    img = Image.new("RGB", (width, height), color="#f8f9fa")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((200, 20), "System Architecture", fill="#333333")

    def box(x, y, w, h, label, color="#4A90D9"):
        draw.rectangle([x, y, x+w, y+h], fill=color, outline="#333", width=2)
        tx = x + w//2 - len(label)*3
        draw.text((tx, y + h//2 - 7), label, fill="white")

    def arrow(x1, y1, x2, y2):
        draw.line([x1, y1, x2, y2], fill="#555", width=2)
        draw.polygon([(x2, y2), (x2-8, y2-5), (x2-8, y2+5)], fill="#555")

    # Boxes
    box(50, 170, 100, 50, "Client")
    box(210, 100, 120, 50, "API Gateway", "#E67E22")
    box(210, 230, 120, 50, "Auth Service", "#9B59B6")
    box(390, 100, 120, 50, "App Server", "#2ECC71")
    box(390, 230, 120, 50, "Database", "#E74C3C")

    # Arrows
    arrow(150, 195, 210, 125)
    arrow(150, 195, 210, 255)
    arrow(330, 125, 390, 125)
    arrow(330, 255, 390, 255)
    arrow(450, 150, 450, 230)

    # Labels
    draw.text((155, 140), "HTTPS", fill="#555")
    draw.text((155, 230), "JWT", fill="#555")
    draw.text((345, 105), "Route", fill="#555")
    draw.text((345, 235), "Query", fill="#555")

    return img


def make_text_image() -> "Image.Image":
    """Create an image containing text (OCR test)."""
    width, height = 500, 300
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    lines = [
        "Invoice #INV-2024-0047",
        "",
        "Bill To: Acme Corporation",
        "Date: March 15, 2024",
        "Due: April 15, 2024",
        "",
        "Services: AI Consulting    $5,400",
        "Software License           $1,200",
        "Support (3 months)           $900",
        "",
        "TOTAL DUE: $7,500.00",
    ]

    y = 20
    for line in lines:
        if line:
            weight = "bold" if line.startswith(("Invoice", "TOTAL", "Bill")) else "normal"
            color = "#c0392b" if "TOTAL" in line else "black"
            draw.text((30, y), line, fill=color)
        y += 22

    return img


def img_to_bytes(img: "Image.Image") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Queries ──────────────────────────────────────────────────────────────────

DEMOS = [
    {
        "name": "Chart reading",
        "make_image": make_chart_image,
        "questions": [
            "What does this chart show? Which month had the highest sales?",
            "What is the approximate total revenue for Q1? Show your calculation.",
        ]
    },
    {
        "name": "Architecture diagram",
        "make_image": make_diagram_image,
        "questions": [
            "Describe the system architecture shown in this diagram.",
            "What security concerns do you see in this architecture?",
        ]
    },
    {
        "name": "Invoice OCR",
        "make_image": make_text_image,
        "questions": [
            "Extract all financial information from this invoice.",
            "What is due and when? Who is it billed to?",
        ]
    },
]


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("=== GEMINI MULTIMODAL DEMO ===\n")

if not LIVE:
    print("Multimodal API shapes:\n")
    print("""
import google.generativeai as genai
from PIL import Image
import io, httpx

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 1. PIL Image object (easiest)
img = Image.open("chart.png")
response = model.generate_content(["Describe this chart:", img])

# 2. Raw bytes
with open("diagram.png", "rb") as f:
    image_bytes = f.read()
response = model.generate_content([
    "Analyze this architecture:",
    {"mime_type": "image/png", "data": image_bytes}
])

# 3. Multiple images + text
response = model.generate_content([
    "Compare these diagrams:",
    img1,
    img2,
    "Which is easier to understand?"
])

# 4. Cross-modal reasoning
response = model.generate_content([
    "Given this invoice image:",
    invoice_img,
    "And this payment policy: invoices over $5000 need manager approval.",
    "Does this invoice need approval?"
])
""")
    print("Token costs for images (Gemini 1.5):")
    print("  Image (any resolution): ~258 tokens flat")
    print("  Audio: ~25 tokens per second")
    print("  Video: ~263 tokens per second")
    print()
    print("Use cases:")
    print("  - Document / invoice processing")
    print("  - Chart / graph reading")
    print("  - Architecture diagram analysis")
    print("  - Screenshot → code generation")
    print("  - Visual QA for product images")
elif not PIL_AVAILABLE:
    print("Install Pillow to run this demo: pip install Pillow")
else:
    model = genai.GenerativeModel(MODEL)

    for demo in DEMOS:
        print(f"=== {demo['name']} ===\n")
        img = demo["make_image"]()
        img_bytes = img_to_bytes(img)
        img_part = {"mime_type": "image/png", "data": img_bytes}

        for question in demo["questions"]:
            print(f"Q: {question}")
            response = model.generate_content([question, img_part])
            print(f"A: {response.text.strip()}\n")
