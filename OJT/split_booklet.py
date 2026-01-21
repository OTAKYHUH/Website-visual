from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent  # folder where split_booklet.py is
HTML_PATH = BASE_DIR / "YSS Training Booklet.html"   # file next to the script
OUT_DIR = BASE_DIR / "yss_booklet" / "pages"

PAGE_BREAK_PATTERN = re.compile(r"<hr[^>]*class=['\"]page-break['\"][^>]*\/?>", re.I)

def main():
    html_text = HTML_PATH.read_text(encoding="utf-8")

    # If it's a full HTML doc, try to extract just <body>...</body>
    body_match = re.search(r"<body[^>]*>(.*)</body>", html_text, flags=re.I | re.S)
    body = body_match.group(1) if body_match else html_text

    # Split into pages
    chunks = PAGE_BREAK_PATTERN.split(body)
    chunks = [c.strip() for c in chunks if c.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):
        out = OUT_DIR / f"{i:03d}.html"
        out.write_text(chunk, encoding="utf-8")
        print("Wrote", out)

    print(f"Done. Pages: {len(chunks)}")

if __name__ == "__main__":
    main()
