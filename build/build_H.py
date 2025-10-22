# build_H.py — Hourly + HSL bundle (Dates+Time DateTime, Shift Date everywhere, column preview)
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

# ===================== small helpers =====================
def _norm(s): 
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())

def _find_col_ci(df: pd.DataFrame, *cands: str) -> str | None:
    """Case/format-insensitive column finder."""
    if df is None or df.empty:
        return None
    idx = {_norm(c): c for c in df.columns}
    for c in cands:
        k = _norm(c)
        if k in idx:
            return idx[k]
    return None

def _map_shift_text(x) -> str:
    s = str(x).strip()
    if s == "1": return "D"
    if s == "2": return "N"
    if s.upper() == "G": return "N"
    return s

def _ensure_shift_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Shift" not in df.columns:
        df["Shift"] = ""
    df["Shift"] = df["Shift"].map(_map_shift_text)
    return df

def _ensure_shift_date(df: pd.DataFrame, *date_candidates: str) -> pd.DataFrame:
    """
    Guarantees a 'Shift Date' column in the format 'dd MMM yy S' (e.g., '02 Sep 25 D').
    If not found, computes it using the first available date-like column.
    """
    df = _ensure_shift_col(df)
    df = df.copy()

    date_col = None
    for c in date_candidates:
        hit = _find_col_ci(df, c)
        if hit:
            date_col = hit
            break
    if date_col is None:
        for c in df.columns:
            if re.search(r"date|dt", str(c), re.I):
                date_col = c
                break
    if date_col is None:
        df["Shift Date"] = ""
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["Shift Date"] = dt.dt.strftime("%d %b %y") + " " + df["Shift"].astype(str).fillna("")
    return df

def _rename_eq_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename EQMT -> Mounting, EQOF -> Offloading (case-insensitive)."""
    lut = {c: _norm(c) for c in df.columns}
    rename_map = {}
    for c, k in lut.items():
        if k in {"eqmt","eqmteu","eqmtteu"}:
            rename_map[c] = "Mounting"
        elif k in {"eqof","eqofeu","eqofteu"}:
            rename_map[c] = "Offloading"
    return df.rename(columns=rename_map)

def _add_datetime_column_from_dates_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build DateTime from **Dates + Time** (fallback: Date or START_DT + Time; else just the date).
    """
    df = df.copy()
    date_col = (
        _find_col_ci(df, "Dates")
        or _find_col_ci(df, "Date")
        or _find_col_ci(df, "START_DT", "START DT", "Start Date")
        or df.columns[0]
    )
    time_col = _find_col_ci(df, "Time", "Hour", "EventTime")

    dts = pd.to_datetime(df[date_col], errors="coerce")
    if time_col:
        d_str = dts.dt.strftime("%Y-%m-%d")
        # FIX: use .str.strip() on Series
        t_str = df[time_col].astype(str).str.strip()
        df["DateTime"] = pd.to_datetime(d_str + " " + t_str, errors="coerce")
    else:
        df["DateTime"] = dts
    return df


def _print_cols(df: pd.DataFrame, name: str, max_width: int = 160) -> None:
    """Pretty-print the column names for a DataFrame."""
    cols = [str(c) for c in df.columns]
    col_line = ", ".join(cols)
    if len(col_line) > max_width:
        col_line = col_line[: max_width - 3] + "..."
    print(f"   • {name}: {len(cols)} columns → [{col_line}]")

# ===================== LOADERS =======================
def read_hourly_sheet(xlsx_path: Path = ALL_XLSX) -> pd.DataFrame:
    """Read 'Hourly' from All.xlsx (raw)."""
    return pd.read_excel(xlsx_path, sheet_name="Hourly")

def read_hsl_logs() -> dict[str, pd.DataFrame]:
    """Load HSL Log & Details, returning dict with whatever is available."""
    out = {}
    x = None
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

# ================= Transform pipelines =================
def add_month_and_shiftdate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly:
      - DateTime from Dates+Time
      - Shift normalized (1->D, 2->N, G->N)
      - Shift Date in 'dd MMM yy S' (using Dates if present)
      - Month (from same base date)
      - Rename EQMT→Mounting, EQOF→Offloading
    """
    df = df.copy()
    df = _add_datetime_column_from_dates_time(df)

    # Shift + Shift Date (prefer Dates, else Date/START_DT)
    df = _ensure_shift_col(df)
    df = _ensure_shift_date(df, "Dates", "Date", "START_DT", "START DT", "Start Date")

    # Month from the same base date as Shift Date (prefer Dates)
    base_date_col = (
        _find_col_ci(df, "Dates")
        or _find_col_ci(df, "Date")
        or _find_col_ci(df, "START_DT", "START DT", "Start Date")
    )
    if base_date_col:
        base_dt = pd.to_datetime(df[base_date_col], errors="coerce")
        df["Month"] = base_dt.dt.strftime("%b")

    df = _rename_eq_cols(df)
    return df

def transform_hsl_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    HSL Log:
      - bucket headers → Good/Average/Poor/Failed
      - drop 'Name'
      - map Shift (1->D, 2->N, G->N)
      - ensure 'Shift Date' (dd MMM yy S) using a date-like column
    """
    df = df.copy()
    # Rename bucket columns
    rename_map = {}
    for col in list(df.columns):
        c_norm = re.sub(r"\s+", "", str(col)).lower()
        if c_norm in {"<=30","le31","lt31","under31"}:
            rename_map[col] = "Good"
        elif c_norm in {"31-60","31to60","31_60","b313060"}:
            rename_map[col] = "Average"
        elif c_norm in {"61-120","61to120","61_120","b61120"}:
            rename_map[col] = "Poor"
        elif re.fullmatch(r">120", c_norm) or re.fullmatch(r"120\+", c_norm) or re.fullmatch(r"over120", c_norm, flags=re.IGNORECASE):
            rename_map[col] = "Failed"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Name" in df.columns:
        df = df.drop(columns=["Name"])

    df = _ensure_shift_col(df)
    df = _ensure_shift_date(df, "Date", "Dates", "DT", "Event Date", "Activity Date")
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
    HSL Details:
      - map Shift (1->D, 2->N, G->N)
      - ensure 'Shift Date' (dd MMM yy S) using 'Activity Date' if available
      - extract 'Start Time' & 'End time'
    """
    df = df.copy()
    # Shift normalization
    shift_col = None
    for cand in ["Shift","shift"]:
        if cand in df.columns:
            shift_col = cand
            break
    if shift_col is None:
        df["Shift"] = ""
        shift_col = "Shift"
    df[shift_col] = df[shift_col].apply(_map_shift_text).astype(str)

    # Shift Date
    df = _ensure_shift_date(df, "Activity Date", "Date", "Dates")

    # Times from datetime-like columns
    act_col = _find_col(df, "Activity Date", "activity date")
    gi_col  = _find_col(df, "Gate In/Prev Trans", "Gate In / Prev Trans", "Gate In")
    dt1 = pd.to_datetime(df[act_col], errors="coerce")
    dt2 = pd.to_datetime(df[gi_col], errors="coerce")
    df["End time"]   = dt1.dt.strftime("%H:%M:%S")
    df["Start Time"] = dt2.dt.strftime("%H:%M:%S")
    return df

# ------------------ Public API ------------------
__all__ = [
    "read_hourly_sheet",
    "read_hsl_logs",
    "add_month_and_shiftdate_hourly",
    "transform_hsl_log",
    "transform_hsl_details",
    "get_tables",
]

def get_tables(xlsx_path: Path = ALL_XLSX, verbose: bool = False) -> dict[str, pd.DataFrame]:
    """
    Returns transformed tables:
      - Hourly  → DateTime (Dates+Time), Shift, **Shift Date (dd MMM yy S)**, Mounting/Offloading, Month
      - Log     → Shift, **Shift Date (dd MMM yy S)**, bucket headers normalized
      - Details → Shift, **Shift Date (dd MMM yy S)**, Start/End times
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

    if verbose:
        print("\n[Table columns]")
        for k, v in out.items():
            if isinstance(v, pd.DataFrame):
                _print_cols(v, k)

    return out

if __name__ == "__main__":
    tables = get_tables(verbose=True)
    print()
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            print(f"[i] {k}: shape={v.shape}")
