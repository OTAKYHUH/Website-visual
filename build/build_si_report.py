#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Build SI Report from all PDFs under static/SI and expose it like other build modules.

- Scans: Path(__file__).resolve().parents[1] / "static" / "SI"
- Output Excel: .../static/Automated Data/SI Report.xlsx
- Appends and drops duplicates (by Filename + Name; Name dedup ignores case/spaces)
- Shift is computed from the HH:MM on the "Inspected on ..." line
- Shift Date is taken from the SAME "Inspected on ..." line (the date before the time),
  formatted as ddmmyyyy (e.g., 01092025) – NOT from the filename.

API (like your other builders): get_tables() -> {"SI Report": DataFrame}
"""

from __future__ import annotations
from pathlib import Path
from datetime import time as dtime
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

def parse_shift_and_date(lines: list[str]) -> tuple[str | None, str | None]:
    """
    Extract HH:MM and the date that precedes it from the 'Inspected on' text.
    Handles compact 'Inspectedon:01Sep202511:57' and spaced variants.
    Returns (shift, shift_date_ddmmyyyy)
    """
    joined = " ".join(lines)

    # Capture date + time in either compact or spaced form
    # e.g., 01Sep2025 11:57  or  01 Sep 2025 11:57
    pat = re.compile(
        r"Inspected\s*on\s*:?\s*.*?\b(?P<d1>\d{2}\s*[A-Za-z]{3}\s*\d{4})\s*(?P<t1>\d{2}:\d{2})",
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
        return None, None

    hh, mm = m.group("t1").split(":")
    t = dtime(int(hh), int(mm))
    shift = "N" if (t >= dtime(19, 30) or t < dtime(7, 30)) else "D"

    # Normalize date token into ddmmyyyy
    dtoken = re.sub(r"\s+", "", m.group("d1"))  # 01Sep2025 or 01Sep2025 (no spaces)
    # Map month
    mon_map = {
        "JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
        "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12",
    }
    m2 = re.match(r"(?P<dd>\d{2})(?P<mon>[A-Za-z]{3})(?P<yyyy>\d{4})", dtoken)
    shift_date = None
    if m2:
        dd = m2.group("dd")
        mm2 = mon_map.get(m2.group("mon").upper(), None)
        yyyy = m2.group("yyyy")
        if mm2:
            shift_date = f"{dd}{mm2}{yyyy}"

    return shift, shift_date

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

# ---------- one PDF -> DataFrame ----------
def process_pdf(pdf_path: Path) -> pd.DataFrame:
    lines = _page1_lines(pdf_path)
    report_number = parse_report_number(lines)
    shift, shift_date = parse_shift_and_date(lines)
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
