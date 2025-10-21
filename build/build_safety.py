#!/usr/bin/env python3
"""
build_demand.py

Reads inputs from the same folder:
- Safety.xlsx       (read to memory only)
- Demand.xlsx       (source for Demand tables)
- All.xlsx sheet "terminal"  (lookup table that maps equipment type → terminal)

Produces two Demand DataFrames in memory:
- Demand        → transformed (unpivot + merge terminal + enrich)
- Demand_2      → a duplicate of the *original* Demand.xlsx sheet (no transforms)

You can also call `read_safety_raw()` to get the Safety sheet as a DataFrame.

No files are written; everything stays in memory.
"""
from pathlib import Path
import pandas as pd
import re

# ========= PATHS =========
FOLDER = Path(__file__).resolve().parents[1] / "static" 
ALL_XLSX = FOLDER / "All.xlsx"
DEMAND_XLSX = FOLDER / "Automated Data/Demand.xlsx"
SAFETY_XLSX = FOLDER / "Automated Data/Safety.xlsx"

# ========= File helpers =========
def _copy_to_temp(p: Path) -> Path:
    import tempfile, shutil
    td = Path(tempfile.mkdtemp(prefix="xlsx_tmp_"))
    dst = td / p.name
    shutil.copy2(p, dst)
    return dst

def _read_first_sheet(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)

def _read_sheet_case_insensitive(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    for s in xl.sheet_names:
        if s.strip().casefold() == sheet_name.strip().casefold():
            return xl.parse(s)
    raise KeyError(f"Sheet {sheet_name!r} not found in {xlsx_path}")

def _tidy_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\u00A0"," ")).strip() for c in df.columns]
    return df

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a lowercase/space-trimmed header mapping for robust column access."""
    mapping = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=mapping)
    return df

def _get_col(df: pd.DataFrame, *candidates: str) -> str:
    """Find a column by trying several case-insensitive names; returns the actual column name."""
    norm = {str(c).strip().casefold(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().casefold()
        if key in norm:
            return norm[key]
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

def _drop_columns_casefold(df: pd.DataFrame, *names: str) -> pd.DataFrame:
    names_cf = {n.casefold() for n in names}
    keep = [c for c in df.columns if str(c).casefold() not in names_cf]
    return df.loc[:, keep]

# ========= Load lookups =========
def read_terminal_lookup(all_xlsx: Path = ALL_XLSX) -> pd.DataFrame:
    df = _read_sheet_case_insensitive(all_xlsx, "terminal")
    df = _tidy_cols(df)
    # normalize header names
    if "equipment type" in {c.lower() for c in df.columns}:
        pass
    return df

# ========= Demand =========
def read_demand_raw(demand_xlsx: Path = DEMAND_XLSX) -> pd.DataFrame:
    df = _read_first_sheet(demand_xlsx)
    return _tidy_cols(df)

def read_safety_raw(safety_xlsx: Path = SAFETY_XLSX) -> pd.DataFrame:
    df = _read_first_sheet(safety_xlsx)
    df = _tidy_cols(df)
    # remove 'Key' column if present
    df = _drop_columns_casefold(df, "Key")
    return df

def build_demand_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Demand: unpivot all columns except Date, Equipment type, Shift
      + merge terminal on Equipment type
      + Month='mmm' from Date
      + rename 'variable'->'label', 'value'->'No.'
      + add Shift Date: dd mmm yy + Shift
      + Change Shift 'G'->'N'
    - Demand_2: duplicate raw sheet, merge terminal, add Month + Shift Date (same as Demand)
    """
    demand_raw = read_demand_raw()
    term = read_terminal_lookup()

    # Normalize & map shift
    demand_raw = _normalize_headers(demand_raw)
    date_col  = _get_col(demand_raw, "Date")
    equip_col = _get_col(demand_raw, "Equipment type", "Equipment", "equipment type")
    shift_col = _get_col(demand_raw, "Shift")
    demand_raw[shift_col] = demand_raw[shift_col].astype(str).str.strip().replace({"G":"N"})

    # Demand_2 = duplicate *original* sheet enriched
    Demand_2 = demand_raw.copy()

    # Unpivot everything except key cols
    id_vars = [date_col, equip_col, shift_col]
    value_vars = [c for c in demand_raw.columns if c not in id_vars]
    Demand = demand_raw.melt(id_vars=id_vars, value_vars=value_vars, var_name="variable", value_name="value")

    # Merge terminal by equipment type
    # Try to find matching column in terminal
    term_cols_cf = {c.casefold(): c for c in term.columns}
    term_equip = term_cols_cf.get("equipment type", None) or term_cols_cf.get("equipment", None) or list(term.columns)[0]
    Demand = Demand.merge(term, left_on=equip_col, right_on=term_equip, how="left")
    Demand_2 = Demand_2.merge(term, left_on=equip_col, right_on=term_equip, how="left")

    # Month + Shift Date
    for df in (Demand, Demand_2):
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["Month"] = dt.dt.strftime("%b")
        df["Shift Date"] = dt.dt.strftime("%d ") + df["Month"].fillna("") + dt.dt.strftime(" %y") + " " + df[shift_col].astype(str)

    # Rename columns variable/value
    Demand = Demand.rename(columns={"variable": "label", "value": "No."})

    # Extend/expand terminal columns: we simply keep whatever columns came from terminal sheet (already merged)

    # Return
    # Reorder a bit for convenience
    demand_cols = [c for c in ["Date","Equipment type","Shift","Month","Shift Date","label","No."] if c in Demand.columns]
    Demand = Demand.reindex(columns=demand_cols + [c for c in Demand.columns if c not in demand_cols])

    d2_cols = [c for c in ["Date","Equipment type","Shift","Month","Shift Date"] if c in Demand_2.columns]
    Demand_2 = Demand_2.reindex(columns=d2_cols + [c for c in Demand_2.columns if c not in d2_cols])

    # Normalize pretty column names for key trio if slightly different
    if date_col != "Date":
        Demand = Demand.rename(columns={date_col: "Date"})
        Demand_2 = Demand_2.rename(columns={date_col: "Date"})
    if equip_col != "Equipment type":
        Demand = Demand.rename(columns={equip_col: "Equipment type"})
        Demand_2 = Demand_2.rename(columns={equip_col: "Equipment type"})
    if shift_col != "Shift":
        Demand = Demand.rename(columns={shift_col: "Shift"})
        Demand_2 = Demand_2.rename(columns={shift_col: "Shift"})

    # Ensure Month based on 'Date' for both
    dt1 = pd.to_datetime(Demand['Date'], errors='coerce')
    Demand['Month'] = dt1.dt.strftime('%b')
    Demand['Shift Date'] = (
        dt1.dt.strftime('%d ') + Demand['Month'].fillna('') + dt1.dt.strftime(' %y') + ' ' + Demand['Shift'].astype(str)
    )

    dt2 = pd.to_datetime(Demand_2['Date'], errors='coerce')
    Demand_2['Month'] = dt2.dt.strftime('%b')
    Demand_2['Shift Date'] = (
        dt2.dt.strftime('%d ') + Demand_2['Month'].fillna('') + dt2.dt.strftime(' %y') + ' ' + Demand_2[shift_col].astype(str)
    )

    return Demand, Demand_2

def main():
    # Build Demand and Demand_2
    demand, demand_2 = build_demand_tables()
    print("Demand (transformed)", demand.shape)
    print(demand.head(12).to_string(index=False))
    print("\nDemand_2 (enriched)", demand_2.shape)
    print(demand_2.head(5).to_string(index=False))

    # Also read Safety.xlsx to memory (no saving) and drop 'Key'
    safety = read_safety_raw()
    print("\nSafety (raw, 'Key' removed if present)", safety.shape)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(safety.head(5).to_string(index=False))

# ------------------ Public API ------------------
__all__ = [
    "read_terminal_lookup",
    "read_demand_raw",
    "read_safety_raw",
    "build_demand_tables",
    "get_tables",
]

def get_tables() -> dict[str, pd.DataFrame]:
    """
    Returns:
      - Demand       (unpivot + terminal merge + Month + 'Shift Date' + renames)
      - Demand_2     (duplicate enriched with terminal + Month + 'Shift Date')
      - Safety       (raw first sheet, 'Key' column removed if present)
      - Terminal     (lookup from All.xlsx 'terminal')
    """
    out: dict[str, pd.DataFrame] = {}

    # Demand pair
    demand, demand_2 = build_demand_tables()
    out["Demand"] = demand
    out["Demand_2"] = demand_2

    # Safety raw (no save)
    try:
        out["Safety"] = read_safety_raw()
    except Exception:
        pass

    # Terminal lookup
    try:
        out["Terminal"] = read_terminal_lookup()
    except Exception:
        pass

    return out

def _cli_preview():
    tables = get_tables()
    for k, v in tables.items():
        print(f"[i] {k}: {v.shape}")

if __name__ == "__main__":
    _cli_preview()
