#!/usr/bin/env python
# -*- coding: utf-8 -*-
# build_attendance.py — reusable (importable) + runnable (HTML preview)

"""
Read the attendance Excel and produce per-person numbers:
- month MC / UL / Turnout
- whole-year totals (or sheet totals)
- pretty HTML preview (like your screenshot)

Pattern is the same as build/photos.py:
  - config at top
  - helpers
  - public API: get_tables(), get_person_attendance_stats(...)
  - __main__ can be run directly to open HTML
"""

from __future__ import annotations

from pathlib import Path
import sys
import webbrowser
from datetime import datetime
import pandas as pd

# ================== CONFIG ==================

BASE = Path(__file__).resolve().parents[1]
# change this to your real file
DEFAULT_ATT_PATH = BASE / "static" / "[NEW] YOD MASTER LIST (MC_UL_TURNOUT).xlsx"

# the sheets we expect to exist – we’ll try to load all of them and stack
ATT_SHEETS = ["P1-3 (YOU)", "P1-3 (YOC)", "P4-6"]

# column names in your file
COL_EMP_ID   = "Employee ID"
COL_STAFF_NO = "Staff No"
COL_NAME     = "Staff Name (JO)"

# months we will show in order
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# =====================================================


def _safe_int(x) -> int:
    """Convert to int safely, treating blanks/NaN as 0."""
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _load_attendance_xlsx(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    Load one sheet (old behavior).
    Kept just in case you still want single-sheet loading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attendance file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def _load_all_attendance_xlsx(path: str | Path, sheet_names: list[str]) -> pd.DataFrame:
    """
    New behavior: load ALL attendance sheets and stack into one DataFrame.
    We add a __source_sheet__ column so you still know where each row came from.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attendance file not found: {path}")
    frames = []
    for sh in sheet_names:
        try:
            df_sh = pd.read_excel(path, sheet_name=sh)
            df_sh["__source_sheet__"] = sh
            frames.append(df_sh)
        except Exception:
            # missing sheet – skip
            continue
    if not frames:
        # fallback to first sheet if nothing was read
        return _load_attendance_xlsx(path, sheet_names[0])
    return pd.concat(frames, ignore_index=True, sort=False)


def _find_person_row(
    df: pd.DataFrame,
    staff_no: str | None = None,
    employee_id: str | None = None,
    name: str | None = None,
) -> pd.Series:
    """
    Try to locate one person by staff_no → employee_id → name.
    If nothing is given / found, just return the first row.
    """
    if staff_no and COL_STAFF_NO in df.columns:
        sub = df.loc[df[COL_STAFF_NO] == staff_no]
        if not sub.empty:
            return sub.iloc[0]

    if employee_id and COL_EMP_ID in df.columns:
        sub = df.loc[df[COL_EMP_ID] == employee_id]
        if not sub.empty:
            return sub.iloc[0]

    if name and COL_NAME in df.columns:
        sub = df.loc[df[COL_NAME] == name]
        if not sub.empty:
            return sub.iloc[0]

    # fallback: just first row
    return df.iloc[0]


def _month_col_from_row(row: pd.Series, month: str, what: str) -> str | None:
    """
    Given a row (person) and month, find the right column name for MC/UL/Turnout.
    Handles small naming variations like Oct_MC vs Oct_M, etc.
    """
    month = month.strip().title()

    if what == "MC":
        for c in (f"{month}_MC", f"{month}_M"):
            if c in row.index:
                return c
        return None

    if what == "UL":
        for c in (f"{month}_UL", f"{month}_U"):
            if c in row.index:
                return c
        return None

    if what == "TURNOUT":
        for c in (f"{month}_Turnout", f"{month}_Turnou"):
            if c in row.index:
                return c
        # fallback: any column that starts with "<Mon>_Turn"
        prefix = f"{month}_Turn"
        for col in row.index:
            if col.startswith(prefix):
                return col
        return None

    return None


def _sum_across_months(row: pd.Series, what: str) -> int:
    """Sum all month columns for MC/UL/Turnout."""
    total = 0
    for m in MONTHS:
        col = _month_col_from_row(row, m, what)
        if col:
            total += _safe_int(row.get(col, 0))
    return total


def _get_total_for_row(row: pd.Series, what: str) -> int:
    """Return the TOTAL column if present, else sum across months."""
    total_cols = {
        "MC": ["Total_MC", "Total MC", "MC TOTAL", "MC_Total"],
        "UL": ["Total_UL", "Total UL", "UL TOTAL", "UL_Total"],
        "TURNOUT": ["Total_Turnout", "Total Turnout", "Turnout TOTAL", "Turnout_Total"],
    }
    for c in total_cols.get(what, []):
        if c in row.index:
            return _safe_int(row.get(c, 0))
    # else sum across months
    return _sum_across_months(row, what)


def _all_month_numbers(row: pd.Series) -> dict[str, dict[str, int]]:
    """
    Return a dict of:
      {"Jan": {"MC":.., "UL":.., "Turnout":..}, ...}
    so we can render the "Full year" table easily.
    """
    result: dict[str, dict[str, int]] = {}
    for m in MONTHS:
        mc_col = _month_col_from_row(row, m, "MC")
        ul_col = _month_col_from_row(row, m, "UL")
        to_col = _month_col_from_row(row, m, "TURNOUT")
        result[m] = {
            "MC": _safe_int(row.get(mc_col, 0)) if mc_col else 0,
            "UL": _safe_int(row.get(ul_col, 0)) if ul_col else 0,
            "Turnout": _safe_int(row.get(to_col, 0)) if to_col else 0,
        }
    return result


def get_person_attendance_stats(
    df: pd.DataFrame,
    month: str,
    staff_no: str | None = None,
    employee_id: str | None = None,
    name: str | None = None,
) -> dict:
    """
    Core API:
      give me the big DataFrame + tell me which person + which month
      → I give you month numbers + totals + full year
    """
    month = month.strip().title()
    row = _find_person_row(df, staff_no=staff_no, employee_id=employee_id, name=name)

    mc_col = _month_col_from_row(row, month, "MC")
    ul_col = _month_col_from_row(row, month, "UL")
    to_col = _month_col_from_row(row, month, "TURNOUT")

    month_mc = _safe_int(row.get(mc_col, 0)) if mc_col else 0
    month_ul = _safe_int(row.get(ul_col, 0)) if ul_col else 0
    month_turnout = _safe_int(row.get(to_col, 0)) if to_col else 0

    total_mc = _get_total_for_row(row, "MC")
    total_ul = _get_total_for_row(row, "UL")
    total_turnout = _get_total_for_row(row, "TURNOUT")

    yearly = _all_month_numbers(row)

    return {
        "person": {
            "name": row.get(COL_NAME, ""),
            "staff_no": row.get(COL_STAFF_NO, ""),
            "employee_id": row.get(COL_EMP_ID, ""),
            "__source_sheet__": row.get("__source_sheet__", ""),
        },
        "month": month,
        "monthly": {
            "MC": month_mc,
            "UL": month_ul,
            "Turnout": month_turnout,
        },
        "totals": {
            "MC": total_mc,
            "UL": total_ul,
            "Turnout": total_turnout,
        },
        "yearly": yearly,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_tables(path: str | Path = DEFAULT_ATT_PATH) -> dict[str, pd.DataFrame | dict]:
    """
    Helper to match your build/ style: return a dict of useful tables.
    """
    df = _load_all_attendance_xlsx(path, ATT_SHEETS)
    return {"attendance_df": df}


def render_preview_html(stats: dict, out_html: str | Path | None = None) -> Path:
    """
    Build the purple HTML you saw in your screenshot.
    """
    person = stats["person"]
    month = stats["month"]
    monthly = stats["monthly"]
    totals = stats["totals"]
    yearly = stats["yearly"]
    generated_at = stats.get("generated_at", "")

    # build year rows
    year_rows = ""
    for m in MONTHS:
        mrow = yearly.get(m, {})
        year_rows += f"""
        <tr>
          <td>{m}</td>
          <td>{mrow.get('MC', 0)}</td>
          <td>{mrow.get('UL', 0)}</td>
          <td>{mrow.get('Turnout', 0)}</td>
        </tr>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Attendance – {person.get('name','')}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
  background: radial-gradient(circle at top, #683cff 0%, #312084 42%, #160e35 100%);
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif;
  padding: 24px;
}}
.panel {{
  max-width: 1100px;
  margin: 0 auto;
}}
h2 {{
  margin: 0 0 6px;
  font-size: 28px;
  font-weight: 700;
}}
.meta {{
  opacity: .72;
  font-size: 12px;
  margin-bottom: 16px;
}}
.cards {{
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin-bottom: 20px;
}}
.card {{
  background: rgba(8,4,18,.15);
  border: 1px solid rgba(255,255,255,.04);
  border-radius: 20px;
  padding: 14px 16px 10px;
  min-width: 160px;
  box-shadow: 0 8px 20px rgba(0,0,0,.15);
}}
.card .label {{
  font-size: 11px;
  text-transform: uppercase;
  opacity: .62;
  margin-bottom: 4px;
}}
.card .val {{
  font-size: 28px;
  font-weight: 600;
}}
h3 {{
  margin-top: 10px;
  font-size: 14px;
  opacity: .8;
}}
.table-wrap {{
  background: rgba(6,3,14,.33);
  border: 1px solid rgba(255,255,255,.03);
  border-radius: 20px;
  overflow: hidden;
  margin-top: 10px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  min-width: 600px;
}}
thead {{
  background: rgba(0,0,0,.25);
}}
th, td {{
  padding: 8px 14px;
  text-align: left;
  font-size: 13px;
}}
tbody tr:nth-child(even) {{
  background: rgba(0,0,0,.085);
}}
footer {{
  margin-top: 16px;
  font-size: 11px;
  opacity: .4;
}}
</style>
</head>
<body>
  <div class="panel">
    <h2>Attendance – {person.get('name','')}</h2>
    <div class="meta">
      Staff No: {person.get('staff_no','') or '–'} · Emp ID: {person.get('employee_id','') or '–'} · Sheet: {person.get('__source_sheet__','') or '–'}
    </div>
    <div class="cards">
      <div class="card">
        <div class="label">{month.upper()} MC</div>
        <div class="val">{monthly.get('MC',0)}</div>
      </div>
      <div class="card">
        <div class="label">{month.upper()} UL</div>
        <div class="val">{monthly.get('UL',0)}</div>
      </div>
      <div class="card">
        <div class="label">{month.upper()} TURNOUT</div>
        <div class="val">{monthly.get('Turnout',0)}</div>
      </div>
      <div class="card">
        <div class="label">TOTAL MC</div>
        <div class="val">{totals.get('MC',0)}</div>
      </div>
      <div class="card">
        <div class="label">TOTAL UL</div>
        <div class="val">{totals.get('UL',0)}</div>
      </div>
      <div class="card">
        <div class="label">TOTAL TURNOUT</div>
        <div class="val">{totals.get('Turnout',0)}</div>
      </div>
    </div>

    <h3>Full year</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Month</th><th>MC</th><th>UL</th><th>Turnout</th></tr>
        </thead>
        <tbody>
          {year_rows}
        </tbody>
      </table>
    </div>

    <footer>Generated: {generated_at}</footer>
  </div>
</body>
</html>
"""
    out_path = Path(out_html) if out_html else Path(__file__).with_name("attendance_preview.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --------- Optional: file picking helpers ----------

def _pick_attendance_file_interactive() -> Path | None:
    """Try Tk file picker; fall back to console prompt."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Attendance Excel",
            filetypes=[("Excel files", "*.xlsx *.xlsm *.xls"), ("All files", "*.*")]
        )
        root.update()
        root.destroy()
        if path:
            p = Path(path)
            if p.exists():
                return p
    except Exception:
        pass

    try:
        inp = input("Enter path to Attendance Excel: ").strip('"').strip()
        if inp:
            p = Path(inp).expanduser()
            if p.exists():
                return p
    except (EOFError, KeyboardInterrupt):
        return None
    return None


def _auto_guess_attendance_file() -> Path | None:
    """Try the default location first, then some common relatives."""
    try:
        if DEFAULT_ATT_PATH.exists():
            return DEFAULT_ATT_PATH
    except Exception:
        pass

    here = Path(__file__).parent
    candidates = [
        here / "attendance.xlsx",
        here / "Attendance.xlsx",
        Path.cwd() / "attendance.xlsx",
        Path.cwd() / "Attendance.xlsx",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


__all__ = [
    "get_tables",
    "get_person_attendance_stats",
    "DEFAULT_ATT_PATH",
    "ATT_SHEETS",
    "MONTHS",
]


if __name__ == "__main__":
    # Usage:
    #   python build_attendance.py "NAME" Oct
    # or python build_attendance.py
    args = sys.argv[1:]
    name = None
    month = "Oct"   # default; this is why you saw OCT in the screenshot
    if len(args) >= 1:
        name = args[0]
    if len(args) >= 2:
        month = args[1]

    att_path = _auto_guess_attendance_file()
    if att_path is None:
        print("[i] Could not find attendance file automatically.")
        att_path = _pick_attendance_file_interactive()
    if att_path is None:
        print("[!] No attendance file selected. Exiting.")
        sys.exit(1)

    df = _load_all_attendance_xlsx(att_path, ATT_SHEETS)

    # 👇 this is the "print the table to see" part
    print("[i] attendance_df preview:")
    print(df.head(10))
    print("[i] columns:", df.columns.tolist())

    stats = get_person_attendance_stats(df, month=month, name=name)
    picked = stats["person"].get("name") or stats["person"].get("staff_no") or stats["person"].get("employee_id")
    print(f"[i] Showing attendance for: {picked!r} ({month})")
    print("[i] Monthly:", stats["monthly"])

    out_html = render_preview_html(stats)
    print(f"[i] Wrote preview to: {out_html}")
    try:
        webbrowser.open(out_html.as_uri(), new=2)
    except Exception:
        print(f"[i] Open this in your browser: {out_html}")
