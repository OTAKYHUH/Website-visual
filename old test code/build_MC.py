# build/mc_details.py — reusable (importable) + runnable (HTML preview)
"""
Pull the text MC/medical details for each month for ONE person from
sheets like:

  MC Consolidation (P1-3)
  MC Consolidation (P4-6)

Columns look like:
  Employee ID | Staff No | Staff Name | Jan | Feb | Mar | ... | Dec
"""

from __future__ import annotations

from pathlib import Path
import sys
import webbrowser
from datetime import datetime
import pandas as pd

# ================== CONFIG ==================

BASE = Path(__file__).resolve().parents[1]
DEFAULT_PATH = BASE / "static" / "[NEW] YOD MASTER LIST (MC_UL_TURNOUT).xlsx"

DETAIL_SHEETS = [
    "MC Consolidation (P1-3)",
    "MC Consolidation (P4-6)",
]

COL_EMP_ID   = "Employee ID"
COL_STAFF_NO = "Staff No"
COL_NAME     = "Staff Name"

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# =====================================================


def _load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Load one sheet, raise if missing."""
    return pd.read_excel(path, sheet_name=sheet_name)


def load_mc_details_df(path: str | Path, sheet_names: list[str]) -> pd.DataFrame:
    """
    Load all MC consolidation sheets and stack into one table.
    """
    path = Path(path)
    frames = []
    for sh in sheet_names:
        try:
            df_sh = pd.read_excel(path, sheet_name=sh)
            df_sh["__source_sheet__"] = sh
            frames.append(df_sh)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No MC consolidation sheets found in: {path}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _find_person_row(
    df: pd.DataFrame,
    staff_no: str | None = None,
    employee_id: str | None = None,
    name: str | None = None,
) -> pd.Series:
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
    return df.iloc[0]


# helper to split multi-line cells into a list
def _split_lines(val) -> list[str]:
    if val is None:
        return []
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    # split on newline OR semicolon
    parts = []
    for line in s.replace(";", "\n").splitlines():
        line = line.strip()
        if line:
            parts.append(line)
    return parts


def _all_month_details(row: pd.Series) -> dict[str, list[str]]:
    """
    Return dict: { 'Jan': ['abc','def'], 'Feb': [...] }
    using the split helper above.
    """
    out: dict[str, list[str]] = {}
    for m in MONTH_COLS:
        if m in row.index:
            out[m] = _split_lines(row.get(m, ""))
        else:
            out[m] = []
    return out


def get_person_mc_details(
    df: pd.DataFrame,
    month: str,
    staff_no: str | None = None,
    employee_id: str | None = None,
    name: str | None = None,
) -> dict:
    """
    Return structure like:
    {
      'person': {...},
      'month': 'Feb',
      'details': 'raw string for that month',
      'months': {'Jan':'...', 'Feb':'...'},
      'month_lists': {'Jan': [...], ...}
    }
    """
    row = _find_person_row(df, staff_no=staff_no, employee_id=employee_id, name=name)

    # normalize month
    month = month.strip()
    if month.lower() == "jun":
        month = "June"
    else:
        month = month.title()

    # pick this month's text
    month_detail = ""
    if month in row.index:
        val = row.get(month, "")
        month_detail = "" if pd.isna(val) else str(val)

    # string version for whole year
    all_months: dict[str, str] = {}
    for col in MONTH_COLS:
        if col in row.index:
            v = row.get(col, "")
            all_months[col] = "" if pd.isna(v) else str(v)
        else:
            all_months[col] = ""

    # list version
    month_lists = _all_month_details(row)

    return {
        "person": {
            "name": row.get(COL_NAME, ""),
            "staff_no": row.get(COL_STAFF_NO, ""),
            "employee_id": row.get(COL_EMP_ID, ""),
            "__source_sheet__": row.get("__source_sheet__", ""),
        },
        "month": month,
        "details": month_detail,
        "months": all_months,
        "month_lists": month_lists,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_tables(path: str | Path = DEFAULT_PATH) -> dict[str, pd.DataFrame | dict]:
    """
    Keep same style as other build/ modules.
    """
    df = load_mc_details_df(path, DETAIL_SHEETS)
    return {"mc_details_df": df}


def render_preview_html(details: dict, out_html: str | Path | None = None) -> Path:
    p = details["person"]
    month = details["month"]
    details_text = details["details"] or "–"
    all_months = details["months"]
    generated_at = details.get("generated_at", "")

    # build table head and body
    head_html = ""
    for m in MONTH_COLS:
        head_html += f"<th>{m}</th>\n"

    body_html = ""
    for m in MONTH_COLS:
        txt = all_months.get(m, "") or "–"
        body_html += f"<td>{txt}</td>\n"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MC Consolidation – {p.get('name','')}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
  background: #0B0D16;
  color: #fff;
  font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Arial,sans-serif;
  padding: 24px;
}}
.panel {{
  background: linear-gradient(135deg, #6c3dd8 0%, #2a1645 65%, #23103a 100%);
  border-radius: 20px;
  padding: 16px 16px 8px;
  max-width: 1150px;
}}
.panel h3 {{
  margin: 0 0 12px;
  font-weight: 600;
  font-size: 16px;
}}
.meta {{
  opacity: .7;
  font-size: 12px;
  margin-bottom: 12px;
}}
.current-month {{
  background: rgba(0,0,0,.15);
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 12px;
  padding: 12px 14px 6px;
  margin-bottom: 14px;
}}
.current-month h4 {{
  margin: 0 0 6px;
  font-size: 14px;
}}
.current-month pre {{
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.4;
  font-family: ui-monospace, SFMono-Regular, SFMono-Medium, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}}
table.months {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  background: rgba(0,0,0,.1);
  border: 1px solid rgba(255,255,255,.03);
  border-radius: 12px;
  overflow: hidden;
  font-size: 12px;
  table-layout: fixed;
}}
table.months th, table.months td {{
  padding: 6px 8px;
  border-bottom: 1px solid rgba(255,255,255,.035);
  vertical-align: top;
  word-wrap: break-word;
  width: 150px;
  line-height: 1.35;
  font-weight: 500;
}}
table.months thead {{
  background: rgba(0,0,0,.2);
}}
footer {{
  margin-top: 12px;
  opacity: .5;
  font-size: 11px;
}}
</style>
</head>
<body>
  <div class="panel">
    <h3>MC Consolidation – {p.get('name','')}</h3>
    <div class="meta">
      Staff No: {p.get('staff_no','') or '–'} · Emp ID: {p.get('employee_id','') or '–'} · Sheet: {p.get('__source_sheet__','') or '–'}
    </div>
    <div class="current-month">
      <h4>{month} details</h4>
      <pre>{details_text}</pre>
    </div>
    <div style="overflow-x:auto">
      <table class="months">
        <thead>
          <tr>
            {head_html}
          </tr>
        </thead>
        <tbody>
          <tr>
            {body_html}
          </tr>
        </tbody>
      </table>
    </div>
    <footer>
      Generated: {generated_at}
    </footer>
  </div>
</body>
</html>
"""
    out_path = Path(out_html) if out_html else Path(__file__).with_name("mc_details_preview.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path

# --------- Optional: file picking helpers ----------
def _pick_mc_file_interactive() -> Path | None:
    """Try Tk file picker; fall back to console prompt."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select MC Consolidation Excel",
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
        inp = input("Enter path to MC Consolidation Excel: ").strip('"').strip()
        if inp:
            p = Path(inp).expanduser()
            if p.exists():
                return p
    except (EOFError, KeyboardInterrupt):
        return None
    return None


def _auto_guess_mc_file() -> Path | None:
    """Try the default location first, then common relatives."""
    try:
        if DEFAULT_PATH.exists():
            return DEFAULT_PATH
    except Exception:
        pass

    here = Path(__file__).parent
    candidates = [
        here / "mc_consolidation.xlsx",
        here / "MC Consolidation.xlsx",
        Path.cwd() / "mc_consolidation.xlsx",
        Path.cwd() / "MC Consolidation.xlsx",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


__all__ = [
    "load_mc_details_df",
    "get_person_mc_details",
    "get_tables",
    "DEFAULT_PATH",
    "DETAIL_SHEETS",
]

if __name__ == "__main__":
    # Usage:
    #   python build_mc_details.py "NAME" Feb
    args = sys.argv[1:]
    name = None
    month = "Jan"
    if len(args) >= 1:
        name = args[0]
    if len(args) >= 2:
        month = args[1]

    mc_path = _auto_guess_mc_file()
    if mc_path is None:
        print("[i] Could not find MC consolidation file automatically.")
        mc_path = _pick_mc_file_interactive()
    if mc_path is None:
        print("[!] No MC file selected. Exiting.")
        sys.exit(1)

    df = load_mc_details_df(mc_path, DETAIL_SHEETS)
    details = get_person_mc_details(df, month=month, name=name)

    picked = details["person"].get("name") or details["person"].get("employee_id") or "FIRST ROW"
    print(f"[i] Showing MC details for: {picked!r} ({month})")
    print("[i] Months (list form):", details["months"])

    out_html = render_preview_html(details)
    print(f"[i] Wrote: {out_html}")
    try:
        webbrowser.open(out_html.as_uri(), new=2)
    except Exception:
        print(f"[i] Open this in your browser: {out_html}")
