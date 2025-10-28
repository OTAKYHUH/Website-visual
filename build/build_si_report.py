#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Build SI Report from all PDFs under static/SI and expose it like other build modules.

- Scans: Path(__file__).resolve().parents[1] / "static" / "SI"
- Output Excel: .../static/Automated Data/SI Report.xlsx
- Appends and drops duplicates (by Filename + Name; Name dedup ignores case/spaces)
- Shift is computed from the HH:MM on the "Inspected on ..." line
- Shift Date is taken from the FILENAME: the text AFTER the last underscore "_".
  If Shift == "N", the Shift Date is (filename date - 1 day).
  Shift Date is stored as ddmmyyyy.

API (like your other builders): get_tables() -> {"SI Report": DataFrame}
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, date, time as dtime, timedelta
import re
import pdfplumber
import pandas as pd

# ---------- PATHS ----------
BASE = Path(__file__).resolve().parents[1]
SI_DIR = BASE / "static" / "SI"
OUT_XLSX = BASE / "static" / "Automated Data" / "SI Report.xlsx"

HEADER_LIKE = re.compile(
    r"^(Report\s*Number|Inspection\s*Date/?Time|Workplace|Date\s*:|Page\s*:|HSS\b)",
    re.IGNORECASE,
)

# ---------- pdf helpers ----------
def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _page1_lines(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        text = (pdf.pages[0].extract_text() or "")
    return [_clean_line(x) for x in text.splitlines() if _clean_line(x)]

# ---------- filename date helpers ----------
_MON_MAP = {
    "JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
    "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12",
}

def _to_ddmmyyyy_from_token(token: str) -> str | None:
    """
    Try to parse a date token into ddmmyyyy.
    Supports:
      - ddmmyyyy (with or without separators)
      - ddmmyyyy (digits with any non-digits stripped)
      - 01Sep2025 / 1Sep2025 (case-insensitive; spaces ignored)
    Returns ddmmyyyy or None.
    """
    raw = token.strip()

    # First, try any digits in order: want exactly 8 digits -> ddmmyyyy
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 8:
        dd, mm, yyyy = digits[:2], digits[2:4], digits[4:]
        try:
            # validate
            datetime.strptime(f"{dd}{mm}{yyyy}", "%d%m%Y")
            return f"{dd}{mm}{yyyy}"
        except ValueError:
            pass

    # Try 01Sep2025 or 1Sep2025 (spaces removed)
    compact = re.sub(r"\s+", "", raw)
    m = re.match(r"(?i)^(?P<d>\d{1,2})(?P<mon>[A-Za-z]{3})(?P<y>\d{4})$", compact)
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
    """
    Extract date token from filename stem after the LAST underscore, and convert to ddmmyyyy.
    Example: ABC_123_01092025.pdf -> 01092025
             ABC_1Sep2025.pdf     -> 01092025
    """
    stem = pdf_path.stem
    if "_" not in stem:
        # Take whole stem as token if no underscore
        token = stem
    else:
        token = stem.rsplit("_", 1)[-1]
    return _to_ddmmyyyy_from_token(token)

def adjust_for_night_shift(shift: str | None, ddmmyyyy: str | None) -> str | None:
    """
    If shift == 'N' and date is present, subtract one day; else return original.
    """
    if not ddmmyyyy:
        return None
    if shift != "N":
        return ddmmyyyy
    try:
        dt = datetime.strptime(ddmmyyyy, "%d%m%Y").date()
        dt2 = dt - timedelta(days=1)
        return dt2.strftime("%d%m%Y")
    except ValueError:
        return ddmmyyyy

# ---------- parsing ----------
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
    """
    Determine shift purely from the 'Inspected on ... HH:MM' time.
    Returns 'D' or 'N' (or None if time not found).
    """
    joined = " ".join(lines)

    # Capture time near "Inspected on"
    pat = re.compile(
        r"Inspected\s*on\s*:?.*?\b(?P<d1>\d{1,2}\s*[A-Za-z]{3}\s*\d{4})?\s*(?P<t1>\d{2}:\d{2})",
        re.IGNORECASE,
    )
    m = pat.search(joined)
    if not m:
        # try line-by-line as fallback
        for ln in lines:
            m = pat.search(ln)
            if m:
                break
    if not m:
        return None

    hh, mm = m.group("t1").split(":")
    t = dtime(int(hh), int(mm))
    return "N" if (t >= dtime(19, 30) or t < dtime(7, 30)) else "D"

# ---------- append + de-dupe ----------
def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()

def append_and_dedupe(out_xlsx: Path, new_rows: pd.DataFrame) -> None:
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

    # Keep the latest copy per (Filename, Name)
    combined = combined.drop_duplicates(subset=["_key_file", "_key_name"], keep="last")
    combined = combined[cols]

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as xlw:
        combined.to_excel(xlw, index=False, sheet_name="SI Report")

# ---------- inspectors parsing (unchanged) ----------
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
    # de-dup in order
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

# ---------- one PDF -> DataFrame ----------
def process_pdf(pdf_path: Path) -> pd.DataFrame:
    lines = _page1_lines(pdf_path)
    report_number = parse_report_number(lines)
    shift = parse_shift(lines)

    # New logic: take date from filename (after last "_"), adjust if Night
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

# ---------- public API (like your other build modules) ----------
__all__ = ["build_si_report", "get_tables"]

def build_si_report() -> pd.DataFrame:
    """
    Process all PDFs under static/SI and write/merge Excel.
    Returns the final DataFrame.
    """
    if not SI_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {SI_DIR}")

    dfs = []
    for pdf in sorted(SI_DIR.glob("*.pdf")):  # use .rglob("*.pdf") to recurse
        try:
            dfs.append(process_pdf(pdf))
        except Exception as e:
            print(f"[WARN] Skipping {pdf.name}: {e}")

    if not dfs:
        # If nothing parsed but file exists, just return what's on disk
        if OUT_XLSX.exists():
            return pd.read_excel(OUT_XLSX, dtype=str)
        return pd.DataFrame(columns=["Filename", "Report Number", "Shift", "Name", "Shift Date"])

    df_all = pd.concat(dfs, ignore_index=True)
    append_and_dedupe(OUT_XLSX, df_all)

    # Return what’s on disk (single source of truth)
    return pd.read_excel(OUT_XLSX, dtype=str)

def get_tables() -> dict[str, pd.DataFrame]:
    """
    Mirrors the pattern in your other builders (returns a dict of DataFrames),
    so routes can do: from build.build_si_report import get_tables
                      tables = get_tables(); df = tables["SI Report"]
    """
    # If Excel already exists, just read; else build it first.
    if OUT_XLSX.exists():
        df = pd.read_excel(OUT_XLSX, dtype=str)
    else:
        df = build_si_report()
    return {"SI Report": df}

# ---------- CLI preview ----------
if __name__ == "__main__":
    df = build_si_report()
    print("SI Report:", df.shape)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.head(10).to_string(index=False))
