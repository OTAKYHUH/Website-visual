# read_hourly_and_hsl.py
from pathlib import Path
import pandas as pd
import re

# ======================= PATHS =======================
ALL_XLSX = Path(__file__).resolve().parents[1] / "static" / "All.xlsx"
HSL_DIR  = Path(__file__).resolve().parents[1] / "static" / "Automated Data" / "HSL" 

# Accept either spelling: "exract" (typo) or "extract"
HSL_CANDIDATES = [
    HSL_DIR / "hsl_exract_log.xlsx",
    HSL_DIR / "hsl_extract_log.xlsx",
]

# ===================== LOADERS =======================
def read_hourly_sheet(xlsx_path: Path = ALL_XLSX) -> pd.DataFrame:
    """Read 'Hourly' from All.xlsx."""
    df = pd.read_excel(xlsx_path, sheet_name="Hourly")
    # Normalize columns commonly seen: Dates / START_DT / Shift
    cols = {str(c).strip(): c for c in df.columns}
    # leave raw; enrichment done by add_month_and_shiftdate_hourly()
    return df

def read_hsl_logs() -> dict[str, pd.DataFrame]:
    """Load HSL Log & Details, returning dict with whatever is available."""
    out = {}
    for p in HSL_CANDIDATES:
        try:
            x = pd.ExcelFile(p)
            break
        except Exception:
            x = None
    if x is None:
        return out
    for sheet in x.sheet_names:
        if sheet.strip().lower() in {"log", "details"}:
            out[sheet.strip().title()] = x.parse(sheet_name=sheet)
    return out

# ================= Transform helpers =================
def _map_shift_text(x) -> str:
    s = str(x).strip()
    if s == "1": return "D"
    if s == "2": return "N"
    if s.upper() == "G": return "N"
    return s

def add_month_and_shiftdate_from_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = dt.dt.strftime("%b")
    if "Shift" not in df.columns:
        df["Shift"] = ""
    df["Shift"] = df["Shift"].map(_map_shift_text)
    df["Shift Date"] = dt.dt.strftime("%d ") + df["Month"].fillna("") + dt.dt.strftime(" %y") + " " + df["Shift"].astype(str)
    return df

def add_month_and_shiftdate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Hourly, create Month='mmm' from Dates (or START_DT if present),
    and 'Shift Date' string based on START_DT + Shift.
    """
    df = df.copy()
    # Use START_DT if present; else Dates
    candidates = [c for c in df.columns if str(c).strip().lower() in {"start_dt","start date","dates","date"}]
    date_col = candidates[0] if candidates else df.columns[0]
    # Try to create Month + Shift Date
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = dt.dt.strftime("%b")
    # Shift handling
    if "Shift" in df.columns:
        df["Shift"] = df["Shift"].map(_map_shift_text)
    else:
        df["Shift"] = ""
    df["Shift Date"] = dt.dt.strftime("%d ") + df["Month"].fillna("") + dt.dt.strftime(" %y") + " " + df["Shift"].astype(str)
    return df

def transform_hsl_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename time bucket columns:
      '< 31' -> 'Good', '31-60' -> 'Average', '61-120' -> 'Poor', '>120' -> 'Failed'
    - Drop 'Name'
    - Shift 1->D, 2->N
    - Add Month & Shift Date (same as Hourly)
    - Rename 'Shift Date' to 'date'
    """
    df = df.copy()
    # Normalize headers
    rename_map = {}
    for col in list(df.columns):
        c_norm = re.sub(r"\s+", "", str(col)).lower()
        if c_norm in {"<31","le31","lt31","under31"}:
            rename_map[col] = "Good"
        elif c_norm in {"31-60","31to60","31_60","b313060"}:
            rename_map[col] = "Average"
        elif c_norm in {"61-120","61to120","61_120","b61120"}:
            rename_map[col] = "Poor"
        elif re.fullmatch(r">120", c_norm) or re.fullmatch(r"120\+", c_norm) or re.fullmatch(r"over120", c_norm, flags=re.IGNORECASE):
            rename_map[col] = "Failed"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop Name
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])

    # Shift mapping
    if "Shift" not in df.columns:
        df["Shift"] = ""
    else:
        df["Shift"] = df["Shift"].apply(_map_shift_text).astype(str)

    # Add Month & Shift Date (using best date-like column)
    date_candidates = [c for c in df.columns if re.search(r"date|dt", str(c), re.I)]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    df = add_month_and_shiftdate_from_col(df, date_col)

    # Rename 'Shift Date' -> 'date'
    if "Shift Date" in df.columns:
        df = df.rename(columns={"Shift Date": "date"})
    return df

def _find_col(df: pd.DataFrame, *names: str) -> str:
    lut = {str(c).strip().casefold(): c for c in df.columns}
    for n in names:
        k = n.strip().casefold()
        if k in lut:
            return lut[k]
    raise KeyError(f"Could not find any of {names} in {list(df.columns)}")

def transform_hsl_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Shift 1->D, 2->N
    - Add Month + 'Shift Date' (same pattern)
    - Duplicate 'Activity Date' -> 'End time' (extract only time)
    - Duplicate 'Gate In/Prev Trans' -> 'Start Time' (extract only time)
    """
    df = df.copy()

    # Shift mapping (handles both "Shift" or "shift")
    shift_col = None
    for cand in ["Shift","shift"]:
        if cand in df.columns:
            shift_col = cand
            break
    if shift_col is None:
        df["Shift"] = ""
        shift_col = "Shift"
    df[shift_col] = df[shift_col].apply(_map_shift_text).astype(str)

    # Month & Shift Date (use a date-like column; usually 'Activity Date')
    date_col = None
    for cand in ["Activity Date","activity date","DATE","Date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]
    df = add_month_and_shiftdate_from_col(df, date_col)

    # Duplicate columns & extract only time
    act_col = _find_col(df, "Activity Date", "activity date")
    gi_col  = _find_col(df, "Gate In/Prev Trans", "Gate In / Prev Trans", "Gate In")
    # create End time / Start Time
    dt1 = pd.to_datetime(df[act_col], errors="coerce")
    dt2 = pd.to_datetime(df[gi_col], errors="coerce")
    df["End time"]   = dt1.dt.strftime("%H:%M:%S")
    df["Start Time"] = dt2.dt.strftime("%H:%M:%S")

    return df

# ===================== QUICK-RUN (optional) =====================
if __name__ == "__main__":
    # 1) Hourly + enrich
    Hourly = read_hourly_sheet()
    Hourly = add_month_and_shiftdate_hourly(Hourly)

    # 2) HSL sheets
    HSL_sheets = read_hsl_logs()
    Log = HSL_sheets.get("Log")
    Details = HSL_sheets.get("Details")

    # 3) Transform Log
    if Log is not None:
        Log = transform_hsl_log(Log)

    # 4) Transform Details
    if Details is not None:
        Details = transform_hsl_details(Details)

    # ---- Optional sanity prints (no saving) ----
    print("[i] Hourly shape:", Hourly.shape)
    print("[i] Hourly columns:", list(Hourly.columns))
    print("[i] HSL sheets loaded:", list(HSL_sheets.keys()))
    if Log is not None:
        print("[i] Log shape:", Log.shape)
        print("[i] Log columns:", list(Log.columns))
    if Details is not None:
        print("[i] Details shape:", Details.shape)
        print("[i] Details columns:", list(Details.columns))
        preview_cols = [c for c in ["date","Month","Shift","Shift Date","End time","Start Time"] if c in Details.columns]
        print("[i] Details preview cols:", preview_cols)

# ------------------ Public API ------------------
__all__ = [
    "read_hourly_sheet",
    "read_hsl_logs",
    "add_month_and_shiftdate_from_col",
    "add_month_and_shiftdate_hourly",
    "transform_hsl_log",
    "transform_hsl_details",
    "get_tables",
]

def get_tables(xlsx_path: Path = ALL_XLSX) -> dict[str, pd.DataFrame]:
    """
    Returns transformed tables for Hourly and HSL logs:
      - Hourly: Hourly with Month + 'Shift Date' added
      - Log:    HSL Log transformed (renamed buckets, Shift D/N, Month + 'Shift Date')
      - Details: HSL Details transformed (Shift D/N, Month + 'Shift Date', Start/End time)
    Also returns 'Log_raw' and 'Details_raw' when available.
    """
    out: dict[str, pd.DataFrame] = {}

    # Hourly (enriched)
    Hourly = read_hourly_sheet(xlsx_path)
    Hourly = add_month_and_shiftdate_hourly(Hourly)
    out["Hourly"] = Hourly

    # HSL sheets
    sheets = read_hsl_logs()
    log_raw = sheets.get("Log")
    det_raw = sheets.get("Details")
    if log_raw is not None:
        out["Log_raw"] = log_raw
        out["Log"] = transform_hsl_log(log_raw.copy())
    if det_raw is not None:
        out["Details_raw"] = det_raw
        out["Details"] = transform_hsl_details(det_raw.copy())

    return out

if __name__ == "__main__":
    _all = get_tables()
    for k, v in _all.items():
        print(f"[i] {k}: {getattr(v, 'shape', '<n/a>')}")
