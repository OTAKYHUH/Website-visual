#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build/build_si_report.py — reusable (importable) + runnable (HTML preview)

Matches the structure of build/photos.py:
- Has PREFERRED_FOLDER
- Provides load/build helpers returning DataFrame
- Provides render_* HTML preview
- Provides get_tables() returning dict for routes
- Optional __main__ preview runner

Data prep/parsing logic is preserved from your existing module.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, date, time as dtime, timedelta
import re, sys, webbrowser
import pdfplumber
import pandas as pd

# ====== YOUR FOLDER (same pattern as photos.py) ======
# PREFERRED_FOLDER = Path(r"C:\...\static\SI")
PREFERRED_FOLDER = Path(__file__).resolve().parents[1] / "static" / "SI"
# =====================================================

# Default output (kept same location as before)
DEFAULT_OUT_XLSX = Path(__file__).resolve().parents[1] / "static" / "Automated Data" / "SI Report.xlsx"

HEADER_LIKE = re.compile(
    r"^(Report\s*Number|Inspection\s*Date/?Time|Workplace|Date\s*:|Page\s*:|HSS\b)",
    re.IGNORECASE,
)

# ---------------- util: HTML rendering ----------------
def _df_to_html_table(df: pd.DataFrame, title: str, meta: str) -> str:
    fmt = {}
    table_html = df.to_html(escape=True, index=False, justify="left", classes="grid", formatters=fmt)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{{background:#0f0f10;color:#e6e6e6;font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Arial,sans-serif;margin:24px}}
  h1{{font-size:20px;margin:0 0 16px}}
  .meta{{opacity:.75;margin-bottom:12px}}
  table.grid{{width:100%;border-collapse:collapse;min-width:680px;background:#121213;border:1px solid #25262b;border-radius:12px;overflow:hidden}}
  thead th{{position:sticky;top:0;background:#15161a;padding:10px;border-bottom:1px solid #25262b;text-align:left}}
  tbody td{{padding:10px;border-bottom:1px dashed #24252a;vertical-align:middle;white-space:nowrap}}
  tbody tr:hover{{background:#16171c}}
  .count{{display:inline-block;background:#1e90ff22;border:1px solid #1e90ff55;color:#bcdcff;
          padding:2px 8px;border-radius:999px;font-weight:700;}}
  .note{{margin-top:12px;opacity:.75}}
</style>
</head>
<body>
  <h1>{title} <span class="count">{len(df)} rows</span></h1>
  <div class="meta">{meta}</div>
  {table_html}
  <div class="note">Columns: {", ".join(df.columns)}</div>
</body>
</html>"""
    return html

# --------------- pdf text helpers (unchanged) ---------------
def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _page1_lines(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        text = (pdf.pages[0].extract_text() or "")
    return [_clean_line(x) for x in text.splitlines() if _clean_line(x)]

# --------------- filename date helpers (unchanged) ---------------
_MON_MAP = {
    "JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
    "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12",
}

def _to_ddmmyyyy_from_token(token: str) -> str | None:
    raw = token.strip()
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 8:
        dd, mm, yyyy = digits[:2], digits[2:4], digits[4:]
        try:
            datetime.strptime(f"{dd}{mm}{yyyy}", "%d%m%Y")
            return f"{dd}{mm}{yyyy}"
        except ValueError:
            pass
    compact = re.sub(r"\s+", "", raw)
    m = re.match(r"(?i)^(?P<d>\d{1,2})(?P<mon>[A-Za-z]{{3}})(?P<y>\d{{4}})$", compact)
    if m:
        dd = f"{int(m.group('d')):02d}"
        mm = _MON_MAP.get(m.group("mon").upper())
        yyyy = m.group("y")
        if mm:
            try:
                datetime.strptime(f"{dd}{mm}{yyyy}", "%d%m%Y")
                return f"{dd}{mm}{yyyy}"
            except ValueError:
                pass
    return None

def shift_date_from_filename(pdf_path: Path) -> str | None:
    stem = pdf_path.stem
    token = stem if "_" not in stem else stem.rsplit("_", 1)[-1]
    return _to_ddmmyyyy_from_token(token)

def adjust_for_night_shift(shift: str | None, ddmmyyyy: str | None) -> str | None:
    if not ddmmyyyy:
        return None
    if shift != "N":
        return ddmmyyyy
    try:
        dt = datetime.strptime(ddmmyyyy, "%d%m%Y").date()
        return (dt - timedelta(days=1)).strftime("%d%m%Y")
    except ValueError:
        return ddmmyyyy

# --------------- parsing (unchanged) ---------------
def parse_report_number(lines: list[str]) -> str:
    for ln in lines:
        if re.search(r"\bReport\s*Number\b", ln, re.IGNORECASE):
            if ":" in ln:
                after = ln.split(":", 1)[1].strip()
                return after.split()[0] if after else ""
            m = re.search(r"\bSIS\d+\b", ln)
            return m.group(0) if m else ""
    return ""

def parse_shift(lines: list[str]) -> str | None:
    joined = " ".join(lines)
    pat = re.compile(
        r"Inspected\s*on\s*:?.*?\b(?P<d1>\d{1,2}\s*[A-Za-z]{3}\s*\d{4})?\s*(?P<t1>\d{2}:\d{2})",
        re.IGNORECASE,
    )
    m = pat.search(joined)
    if not m:
        for ln in lines:
            m = pat.search(ln)
            if m:
                break
    if not m:
        return None
    hh, mm = m.group("t1").split(":")
    t = dtime(int(hh), int(mm))
    return "N" if (t >= dtime(19, 30) or t < dtime(7, 30)) else "D"

# --------------- append + de-dupe (unchanged) ---------------
def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()

def _append_and_dedupe(out_xlsx: Path, new_rows: pd.DataFrame) -> None:
    cols = ["Filename", "Report Number", "Shift", "Name", "Shift Date"]
    new_rows = new_rows[cols].copy()
    new_rows["_key_file"] = new_rows["Filename"].astype(str)
    new_rows["_key_name"] = new_rows["Name"].astype(str).map(_norm_name)

    if out_xlsx.exists():
        try:
            existing = pd.read_excel(out_xlsx, dtype=str)
        except Exception:
            existing = pd.DataFrame(columns=cols)
        existing = existing.astype(str)
        existing["_key_file"] = existing["Filename"].astype(str)
        existing["_key_name"] = existing["Name"].astype(str).map(_norm_name)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=["_key_file", "_key_name"], keep="last")
    combined = combined[cols]

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as xlw:
        combined.to_excel(xlw, index=False, sheet_name="SI Report")

# --------------- inspectors parsing (unchanged) ---------------
def _candidate_chunks(text: str) -> list[str]:
    return [c for c in (re.split(r"\s{2,}", text.strip()) or [text.strip()]) if c]

def _tail_has_yod_psa(tail: str) -> bool:
    norm = re.sub(r"\s+", "", tail).upper()
    return (",YOD," in norm) and ("PSA" in norm)

def _extract_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    for c in _candidate_chunks(text):
        if "," not in c:
            continue
        name, tail = c.split(",", 1)
        name, tail = name.strip(), tail.strip()
        if not name or len(name) < 3 or re.search(r"\d", name):
            continue
        if not _tail_has_yod_psa(tail):
            continue
        names.append(name)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def parse_inspectors(lines: list[str]) -> list[str]:
    names: list[str] = []
    patt = re.compile(r"\bInspectors?\s*:\s*(.*)", re.IGNORECASE)
    for i, ln in enumerate(lines):
        m = patt.search(ln)
        if m:
            tail = m.group(1).strip()
            if tail: names.extend(_extract_names_from_text(tail))
            j = i + 1
            while j < len(lines) and len(names) < 12:
                cand = lines[j]
                if HEADER_LIKE.search(cand):
                    break
                extra = _extract_names_from_text(cand)
                if not extra and names:
                    break
                names.extend(extra); j += 1
            break
    return [n.strip() for n in names if n.strip()]

# --------------- one PDF -> DataFrame (unchanged) ---------------
def _process_pdf(pdf_path: Path) -> pd.DataFrame:
    lines = _page1_lines(pdf_path)
    report_number = parse_report_number(lines)
    shift = parse_shift(lines)
    fn_ddmmyyyy = shift_date_from_filename(pdf_path)
    shift_date = adjust_for_night_shift(shift, fn_ddmmyyyy)
    inspectors = parse_inspectors(lines)

    base = {
        "Filename": pdf_path.name,
        "Report Number": report_number,
        "Shift": shift,
        "Shift Date": shift_date,
    }
    rows = [{**base, "Name": n} for n in (inspectors or [""])]
    return pd.DataFrame(rows, columns=["Filename", "Report Number", "Shift", "Name", "Shift Date"])

# ----------------- public helpers (aligned to photos.py) -----------------
def load_si_report(si_dir: str | Path = PREFERRED_FOLDER,
                   out_xlsx: str | Path = DEFAULT_OUT_XLSX,
                   recurse: bool = False) -> pd.DataFrame:
    """
    Scan PDFs in si_dir, append+dedupe into out_xlsx, and return the final DataFrame.
    """
    si_dir = Path(si_dir)
    out_xlsx = Path(out_xlsx)

    if not si_dir.exists():
        raise FileNotFoundError(f"Folder not found: {si_dir}")

    pattern = "**/*.pdf" if recurse else "*.pdf"
    pdfs = sorted(si_dir.glob(pattern))
    dfs: list[pd.DataFrame] = []
    for pdf in pdfs:
        if pdf.is_file() and pdf.suffix.lower() == ".pdf" and not pdf.name.startswith("~$"):
            try:
                dfs.append(_process_pdf(pdf))
            except Exception as e:
                print(f"[WARN] Skipping {pdf.name}: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        _append_and_dedupe(out_xlsx, df_all)

    # Single source of truth: whatever is on disk (create empty if absent)
    if out_xlsx.exists():
        return pd.read_excel(out_xlsx, dtype=str)
    else:
        empty = pd.DataFrame(columns=["Filename", "Report Number", "Shift", "Name", "Shift Date"])
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as xlw:
            empty.to_excel(xlw, index=False, sheet_name="SI Report")
        return empty

def render_si_table(si_dir: str | Path = PREFERRED_FOLDER,
                    out_html: str | Path | None = None,
                    title: str = "SI Report Table",
                    out_xlsx: str | Path = DEFAULT_OUT_XLSX) -> Path:
    """
    Build a simple self-contained HTML preview of the SI Report.
    """
    df = load_si_report(si_dir=si_dir, out_xlsx=out_xlsx)
    meta = f'Folder: <code>{Path(si_dir).resolve()}</code> &nbsp;|&nbsp; Output: <code>{Path(out_xlsx).resolve()}</code>'
    html = _df_to_html_table(df, title=title, meta=meta)
    out_path = Path(out_html) if out_html else Path(__file__).with_name("si_report_preview.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path

def get_tables(si_dir: str | Path = PREFERRED_FOLDER,
               out_xlsx: str | Path = DEFAULT_OUT_XLSX) -> dict[str, pd.DataFrame]:
    """
    Mirrors photos.get_tables:
      return {"SI Report": DataFrame}
    Usage (routes):
      from build.build_si_report import get_tables
      tables = get_tables()             # uses PREFERRED_FOLDER
      df = tables["SI Report"]
    """
    df = load_si_report(si_dir=si_dir, out_xlsx=out_xlsx)
    return {"SI Report": df}

# ----------------- Optional: folder helpers (like photos.py) -----------------
def _iter_pdf_files(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("~$"):
            yield p

def _pick_folder_interactive() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askdirectory(title="Select folder with SI PDFs")
        root.update(); root.destroy()
        if path:
            p = Path(path)
            if p.exists() and p.is_dir():
                return p
    except Exception:
        pass
    try:
        inp = input("Enter folder path that contains SI PDFs: ").strip('"').strip()
        if inp:
            p = Path(inp).expanduser()
            if p.exists() and p.is_dir():
                return p
    except (EOFError, KeyboardInterrupt):
        return None
    return None

def _auto_guess_folder() -> Path | None:
    try:
        if PREFERRED_FOLDER.exists() and PREFERRED_FOLDER.is_dir():
            if any(_iter_pdf_files(PREFERRED_FOLDER)):
                return PREFERRED_FOLDER
    except Exception:
        pass
    here = Path(__file__).parent
    candidates = [
        here / "SI",
        here / "static" / "SI",
        Path.cwd() / "SI",
        Path.cwd() / "static" / "SI",
        Path.cwd(),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir() and any(_iter_pdf_files(cand)):
            return cand
    return None

# ------------------ Public API ------------------
__all__ = [
    "load_si_report",
    "render_si_table",
    "get_tables",
    "PREFERRED_FOLDER",
]

# ------------------ Runnable preview ------------------
if __name__ == "__main__":
    folder = _auto_guess_folder()
    if folder is None:
        print("[i] No PDFs found in your preferred folder or common locations.")
        print("[i] Please pick a folder…")
        folder = _pick_folder_interactive()

    if not folder:
        print("[!] No folder selected. Exiting.")
        sys.exit(1)

    if not any(_iter_pdf_files(folder)):
        print(f"[!] Folder has no PDFs: {folder}")
        sys.exit(2)

    out_file = render_si_table(si_dir=folder, title="SI Report Table (Preview)")
    print(f"[i] Wrote: {out_file}")
    try:
        webbrowser.open(out_file.as_uri(), new=2)
    except Exception:
        print(f"[i] Open this in your browser: {out_file}")
