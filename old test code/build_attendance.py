# build/attendance.py — reusable (importable) + runnable (HTML preview)
"""
Read the attendance Excel and produce per-person numbers:
- month MC / UL / Turnout
- whole-year (or sheet) totals

Supports 3 sheets with the same format:
  "P1-3 (YOU)", "P1-3 (YOC)", "P4-6"
"""

from __future__ import annotations

from pathlib import Path
import sys
import webbrowser
from datetime import datetime
import pandas as pd

# ================== CONFIG ==================

BASE = Path(__file__).resolve().parents[1]
# adjust to your actual file name/location
DEFAULT_ATT_PATH = BASE / "static" / "[NEW] YOD MASTER LIST (MC_UL_TURNOUT).xlsx"

# original single name (kept – we won’t delete anything)
ATT_SHEET_NAME = "P1-3 (YOU)"
# the actual list of sheets we will merge
ATT_SHEETS = ["P1-3 (YOU)", "P1-3 (YOC)", "P4-6"]

COL_EMP_ID   = "Employee ID"
COL_STAFF_NO = "Staff No"
COL_NAME     = "Staff Name (JO)"

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


def _load_attendance_xlsx(path: str | Path, sheet_name: str = ATT_SHEET_NAME) -> pd.DataFrame:
    """
    Old behavior:
      read ONE sheet only, just like your original file.
    Kept for compatibility – loads just ONE sheet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attendance file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def _load_all_attendance_xlsx(path: str | Path, sheet_names: list[str]) -> pd.DataFrame:
    """
    New: load ALL attendance sheets and stack into one table.
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
        # fallback to single-sheet loader
        return _load_attendance_xlsx(path)
    return pd.concat(frames, ignore_index=True, sort=False)


def _find_person_row(
    df: pd.DataFrame,
    staff_no: str | None = None,
    employee_id: str | None = None,
    name: str | None = None,
) -> pd.Series:
    """
    Try to locate one person by staff_no → employee_id → name.
    Fall back to FIRST row if none given.
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
    Handles small naming variations.
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
    # common total patterns
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

    # full year
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
    Wrapper to match your existing pattern: load once, return dict of useful tables.
    """
    df = _load_all_attendance_xlsx(path, ATT_SHEETS)
    return {"attendance_df": df}


def render_preview_html(stats: dict, out_html: str | Path | None = None) -> Path:
    person = stats["person"]
    month = stats["month"]
    monthly = stats["monthly"]
    totals = stats["totals"]
    yearly = stats["yearly"]
    generated_at = stats.get("generated_at", "")

    # build small yearly table
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
<title>Attendance Preview – {person.get('name','')}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
  background: #0F0A1E;
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif;
  padding: 24px;
}}
.panel {{
  background: radial-gradient(circle at top, #5e35ff 0%, #391a65 65%, #26113f 100%);
  border-radius: 16px;
  padding: 20px 24px 28px;
  max-width: 820px;
  box-shadow: 0 10px 35px rgba(0,0,0,.25);
}}
.row {{
  display: flex;
  gap: 32px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}}
.box {{
  background: rgba(15,10,30,.25);
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 12px;
  padding: 10px 14px 8px;
  min-width: 120px;
}}
.box .label {{
  font-size: 11px;
  text-transform: uppercase;
  opacity: .6;
  margin-bottom: 4px;
}}
.box .val {{
  font-size: 20px;
  font-weight: 600;
}}
table.year {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
  background: rgba(9,6,20,.35);
  border: 1px solid rgba(255,255,255,.03);
  border-radius: 12px;
  overflow: hidden;
}}
table.year th, table.year td {{
  padding: 6px 10px;
  text-align: left;
  font-size: 13px;
}}
table.year thead {{
  background: rgba(0,0,0,.25);
}}
footer {{
  margin-top: 16px;
  font-size: 11px;
  opacity: .45;
}}
</style>
</head>
<body>
  <div class="panel">
    <h2 style="margin-top:0;margin-bottom:10px;">Attendance – {person.get('name','')}</h2>
    <div style="opacity:.6;font-size:12px;margin-bottom:12px;">
      Staff No: {person.get('staff_no','') or '–'} · Emp ID: {person.get('employee_id','') or '–'} · Sheet: {person.get('__source_sheet__','') or '–'}
    </div>
    <div class="row">
      <div class="box">
        <div class="label">{month} MC</div>
        <div class="val">{monthly.get('MC',0)}</div>
      </div>
      <div class="box">
        <div class="label">{month} UL</div>
        <div class="val">{monthly.get('UL',0)}</div>
      </div>
      <div class="box">
        <div class="label">{month} Turnout</div>
        <div class="val">{monthly.get('Turnout',0)}</div>
      </div>
      <div class="box">
        <div class="label">Total MC</div>
        <div class="val">{totals.get('MC',0)}</div>
      </div>
      <div class="box">
        <div class="label">Total UL</div>
        <div class="val">{totals.get('UL',0)}</div>
      </div>
      <div class="box">
        <div class="label">Total Turnout</div>
        <div class="val">{totals.get('Turnout',0)}</div>
      </div>
    </div>

    <h3 style="margin-top:18px;margin-bottom:6px;font-size:13px;opacity:.85;">Full year</h3>
    <table class="year">
      <thead>
        <tr><th>Month</th><th>MC</th><th>UL</th><th>Turnout</th></tr>
      </thead>
      <tbody>
        {year_rows}
      </tbody>
    </table>

    <footer>
      Generated: {generated_at}
    </footer>
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
    """Try the default location first, then common relatives."""
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
    "ATT_SHEET_NAME",
    "ATT_SHEETS",
    "MONTHS",
]

if __name__ == "__main__":
    # Usage:
    #   python build_attendance.py "NAME" Oct
    args = sys.argv[1:]
    name = None
    month = "Oct"
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
