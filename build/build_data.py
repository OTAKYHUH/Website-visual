# build_data.py
from pathlib import Path
import pandas as pd
import tempfile, shutil, os, re, calendar

# ========= CONFIG =========
TARGET = Path(__file__).resolve().parents[1] / "static"
GLOB_PATTERN = "**/*.xlsx"

# Sheets
TARGET_SHEET     = "P123 Setup"       # P123 roles (unpivot)
WAIT_SHEET       = "P123 Wait Time"   # P123 YWT
GROUP_SHEET      = "Group"            # Staff group/terminal
YMS_SHEET        = "YMS"              # YMS raw
SS_SHEET         = "SS"               # SS raw
P456_SETUP_SHEET = "P456 Setup"       # P456 roles (unpivot)
P456_WAIT_SHEET  = "P456 Wait Time"   # P456 YWT

# --------- helpers to open local/locked files ---------
def pick_latest_excel(folder: Path, pattern: str) -> Path | None:
    candidates = [
        p for p in folder.glob(pattern)
        if p.is_file() and not p.name.startswith("~$")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _temp_copy(path: Path) -> Path:
    td = Path(tempfile.mkdtemp(prefix="xlsx_"))
    dst = td / path.name
    shutil.copy2(path, dst)
    return dst

def load_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=sheet_name)

def _resolve_source(src: Path | None) -> Path:
    """
    If src is a file, use it.
    If src is a folder (or None), prefer a known master like 'All.xlsx';
    otherwise pick the newest .xlsx that is NOT under 'Automated Data' and not named Safety/Demand.
    """
    def _prefer_master(folder: Path) -> Path | None:
        # Highest priority exact filenames
        for name in ("All.xlsx", "all.xlsx"):
            p = folder / name
            if p.is_file():
                return p
        # Next: any All*.xlsx deeper in the tree
        hits = list(folder.rglob("All*.xlsx"))
        for p in hits:
            if p.is_file() and not p.name.startswith("~$"):
                return p
        return None

    if src is not None:
        src = Path(src)
        if src.is_file():
            return src
        if src.is_dir():
            m = _prefer_master(src)
            if m: return m
            # fall back to most recent xlsx in this folder tree (but avoid Automated Data/Safety/Demand trap)
            candidates = [
                p for p in src.rglob("*.xlsx")
                if p.is_file()
                and not p.name.startswith("~$")
                and "automated data" not in str(p.parent).lower()
                and p.stem.lower() not in {"safety", "demand"}
            ]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(f"Could not resolve excel from: {src}")

    # src is None → resolve from global TARGET
    if TARGET.is_file():
        return TARGET
    if TARGET.is_dir():
        m = _prefer_master(TARGET)
        if m: return m
        # same guarded fallback
        candidates = [
            p for p in TARGET.rglob("*.xlsx")
            if p.is_file()
            and not p.name.startswith("~$")
            and "automated data" not in str(p.parent).lower()
            and p.stem.lower() not in {"safety", "demand"}
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError("Could not resolve source Excel from TARGET.")


# --------- Cleaning utilities ---------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        s = str(c)
        s = s.replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        new_cols.append(s)
    df.columns = new_cols
    return df

def _normkey(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u00A0", " ")).strip().casefold()

def _find_column(df: pd.DataFrame, *candidates: str) -> str:
    lut = { _normkey(c): c for c in df.columns }
    for cand in candidates:
        k = _normkey(cand)
        if k in lut:
            return lut[k]
    raise KeyError(f"Unable to find any of {candidates} in columns: {list(df.columns)}")

def _drop_fully_blank_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = [not df[c].dropna().astype(str).str.strip().eq("").all() for c in df.columns]
    return df.loc[:, mask]

def _clean_name_text(s: pd.Series) -> pd.Series:
    if s.dtype != object:
        s = s.astype(str)
    out = (s.astype(str)
             .str.replace("\u00A0", " ", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())
    out = out.mask(out.str.lower().isin(["-", "--", "nan", "none"]), "")
    out = out.str.replace(r"^irfAN BIN RAHMAN$", "IRFAN BIN RAHMAN", case=False, regex=True)
    return out

# --------- Month helpers ---------
def build_month_lookup() -> pd.DataFrame:
    rows = [{"MonthNum": i,
             "Month": calendar.month_name[i],
             "MonthAbbr": calendar.month_abbr[i],
             "MonthKey": f"{i:02d}"} for i in range(1, 13)]
    return pd.DataFrame(rows)

def add_month_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Adds Month/MonthNum and a 'Shift Date' text column if 'Shift' or 'EVENT_HR12_SHIFT_C'
    exists. Builds 'Shift Date' like: dd mmm yy + ' ' + shift.
    """
    df = df.copy()
    if date_col not in df.columns:
        return df
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["MonthNum"] = dt.dt.month
    df["Month"]    = dt.dt.strftime("%b")  # 'mmm'

    # Prefer 'Shift', else use EVENT_HR12_SHIFT_C
    shift_col = None
    if "Shift" in df.columns:
        shift_col = "Shift"
    elif "EVENT_HR12_SHIFT_C" in df.columns:
        shift_col = "EVENT_HR12_SHIFT_C"

    if shift_col:
        df[shift_col] = df[shift_col].astype(str)
        df["Shift Date"] = (
            dt.dt.strftime("%d ") + df["Month"].fillna("") + dt.dt.strftime(" %y") + " " + df[shift_col]
        )
    return df

# --------- Wait-time reducers (shift-level) ---------
def _first_non_null(s: pd.Series):
    for x in s:
        if pd.notna(x):
            return x
    return pd.NA

def _reduce_wait_to_dateshift(wait_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse wait_df to unique (EVENT_SHIFT_DT, EVENT_HR12_SHIFT_C).
    - Numeric cols: mean (ignore NaN)
    - Non-numeric: first non-null
    """
    if wait_df.empty:
        return wait_df

    ds_keys = ["EVENT_SHIFT_DT", "EVENT_HR12_SHIFT_C"]
    wait_df = wait_df.copy()
    wait_df["EVENT_SHIFT_DT"] = pd.to_datetime(wait_df["EVENT_SHIFT_DT"], errors="coerce")
    wait_df["EVENT_HR12_SHIFT_C"] = wait_df["EVENT_HR12_SHIFT_C"].astype(str).str.strip().replace({"G":"N"})

    num_cols = wait_df.select_dtypes(include="number").columns.difference(ds_keys)
    non_num_cols = wait_df.columns.difference(list(num_cols) + ds_keys)

    agg_map = {c: "mean" for c in num_cols}
    for c in non_num_cols:
        agg_map[c] = _first_non_null

    out = (wait_df
           .groupby(ds_keys, dropna=False, as_index=False)
           .agg(agg_map))
    return out

# --------- P123 ---------
def _coerce_and_clean_setup(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    df = _drop_fully_blank_columns(df)
    # Common columns: DATE / SHIFT / role columns...
    date_col  = _find_column(df, "DATE", "Date")
    shift_col = _find_column(df, "SHIFT", "Shift")
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
    df[shift_col] = df[shift_col].astype(str).str.strip().replace({"G":"N"})
    # Fix role names
    for c in df.columns:
        if c not in [date_col, shift_col]:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def unpivot_p123_setup(df: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_and_clean_setup(df)
    date_col  = _find_column(df, "DATE", "Date")
    shift_col = _find_column(df, "SHIFT", "Shift")
    id_vars   = [date_col, shift_col]
    value_vars = [c for c in df.columns if c not in id_vars]
    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Role", value_name="Name")
    long_df["Name"] = _clean_name_text(long_df["Name"])
    long_df = long_df[(long_df["Name"].notna()) & (long_df["Name"].astype(str).str.strip() != "")]
    long_df = long_df.rename(columns={date_col: "DATE", shift_col: "SHIFT"})
    long_df["SHIFT"] = long_df["SHIFT"].astype(str).replace({"G":"N"})
    long_df["DATE"]  = pd.to_datetime(long_df["DATE"], errors="coerce")
    return long_df

def load_p123_setup_long(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    df = load_sheet(xlsx, TARGET_SHEET)
    long_df = unpivot_p123_setup(df)
    return long_df

def _clean_wait_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Name is NOT required. We keep whatever columns exist, but ensure:
    - EVENT_SHIFT_DT
    - EVENT_HR12_SHIFT_C
    """
    df = _normalize_columns(df)
    date_col  = _find_column(df, "EVENT_SHIFT_DT", "EVENT SHIFT DT", "EVENT_SHIFT_DATE", "DATE")
    shift_col = _find_column(df, "EVENT_HR12_SHIFT_C", "SHIFT", "Event Shift")
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
    df[shift_col] = df[shift_col].astype(str).replace({"G":"N"}).str.strip()
    df = df.rename(columns={
        date_col: "EVENT_SHIFT_DT",
        shift_col: "EVENT_HR12_SHIFT_C",
    })
    return df

def load_wait_time(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    df = load_sheet(xlsx, WAIT_SHEET)
    return _clean_wait_time(df)

def load_p123_with_ywt(src: Path | None = None) -> pd.DataFrame:
    long_df = load_p123_setup_long(src)  # DATE, SHIFT, Role, Name
    ywt     = load_wait_time(src)        # shift-level (after reduction)
    ywt_ds  = _reduce_wait_to_dateshift(ywt)

    left = long_df.rename(columns={"DATE":"EVENT_SHIFT_DT", "SHIFT":"EVENT_HR12_SHIFT_C"}).copy()
    left["EVENT_SHIFT_DT"] = pd.to_datetime(left["EVENT_SHIFT_DT"], errors="coerce")
    left["EVENT_HR12_SHIFT_C"] = left["EVENT_HR12_SHIFT_C"].astype(str).str.replace("G","N").str.strip()

    out = left.merge(
        ywt_ds,
        on=["EVENT_SHIFT_DT","EVENT_HR12_SHIFT_C"],
        how="left",
        suffixes=("","_ywt")
    )
    out = add_month_columns(out, "EVENT_SHIFT_DT")
    return out

# --------- Group & lookups ---------
def load_group_filtered(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    df = load_sheet(xlsx, GROUP_SHEET)
    df = _normalize_columns(df)
    # Keep non-blank group rows
    group_col = _find_column(df, "Group", "GROUP", "Team")
    term_col  = _find_column(df, "terminal", "Terminal")
    name_col  = _find_column(df, "Name", "NAME")
    df[name_col] = _clean_name_text(df[name_col])
    df = df.rename(columns={group_col: "group", term_col: "terminal", name_col: "Name"})
    df = df[(df["group"].astype(str).str.strip() != "") & (df["Name"].astype(str).str.strip() != "")]
    return df

def _build_group_lookup(df_group: pd.DataFrame) -> pd.DataFrame:
    # minimal lookup by Name
    return df_group[["Name","group","terminal"]].drop_duplicates()

def _build_group_lookup_full(src: Path | None = None) -> pd.DataFrame:
    grp = load_group_filtered(src)
    return _build_group_lookup(grp)

def load_yms_filtered(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    df = load_sheet(xlsx, YMS_SHEET)
    df = _normalize_columns(df)
    # Filter blanks
    keep_cols = [c for c in df.columns if not df[c].isna().all()]
    df = df[keep_cols]
    return df

def load_ss_enriched(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    df = load_sheet(xlsx, SS_SHEET)
    df = _normalize_columns(df)
    # Example enrichment: drop fully blank columns & tidy names
    df = _drop_fully_blank_columns(df)
    if "Name" in df.columns:
        df["Name"] = _clean_name_text(df["Name"])
    return df

# --------- P456 ---------
def _clean_p456_setup(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    date_col  = _find_column(df, "DATE", "Date")
    shift_col = _find_column(df, "SHIFT", "Shift")
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
    df[shift_col] = df[shift_col].astype(str).replace({"G":"N"}).str.strip()
    # Melt all others into Role/Name
    role_cols = [c for c in df.columns if c not in [date_col, shift_col]]
    long_df = df.melt(id_vars=[date_col, shift_col], value_vars=role_cols, var_name="Role", value_name="Name")
    long_df["Name"] = _clean_name_text(long_df["Name"])
    long_df = long_df[(long_df["Name"].notna()) & (long_df["Name"].astype(str).str.strip() != "")]
    long_df = long_df.rename(columns={date_col: "DATE", shift_col: "SHIFT"})
    return long_df

def _clean_p456_wait(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    date_col  = _find_column(df, "EVENT_SHIFT_DT", "DATE")
    shift_col = _find_column(df, "EVENT_HR12_SHIFT_C", "SHIFT")
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
    df[shift_col] = df[shift_col].astype(str).replace({"G":"N"}).str.strip()
    df = df.rename(columns={date_col: "EVENT_SHIFT_DT", shift_col: "EVENT_HR12_SHIFT_C"})
    return df

def load_p456_enriched(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    setup = load_sheet(xlsx, P456_SETUP_SHEET)
    wait  = load_sheet(xlsx, P456_WAIT_SHEET)
    setup_long = _clean_p456_setup(setup)
    wait_clean = _clean_p456_wait(wait)
    ywt_ds     = _reduce_wait_to_dateshift(wait_clean)

    left = setup_long.rename(columns={"DATE":"EVENT_SHIFT_DT", "SHIFT":"EVENT_HR12_SHIFT_C"}).copy()
    left["EVENT_SHIFT_DT"] = pd.to_datetime(left["EVENT_SHIFT_DT"], errors="coerce")
    left["EVENT_HR12_SHIFT_C"] = left["EVENT_HR12_SHIFT_C"].astype(str).str.replace("G","N").str.strip()

    out = left.merge(
        ywt_ds,
        on=["EVENT_SHIFT_DT","EVENT_HR12_SHIFT_C"],
        how="left"
    )
    out = add_month_columns(out, "EVENT_SHIFT_DT")
    return out

def load_p456_with_group_enriched(src: Path | None = None) -> pd.DataFrame:
    """
    P456: add group/terminal by Name (left-join), keep blanks if no match.
    """
    p456 = load_p456_enriched(src)
    grp_lu = _build_group_lookup_full(src)   # columns: Name, group, terminal
    out = p456.merge(grp_lu, on="Name", how="left")
    for col in ["group", "terminal"]:
        if col in out.columns:
            out[col] = out[col].fillna("")
    return out

# --------- Enriched & appended ---------
def load_p123_with_ywt_group_enriched(src: Path | None = None) -> pd.DataFrame:
    p123_ywt = load_p123_with_ywt(src)
    grp_lu   = _build_group_lookup_full(src)
    out = p123_ywt.merge(grp_lu, on="Name", how="left")
    # Replace nulls in group/terminal with blanks for easier filters
    for col in ["group","terminal"]:
        if col in out.columns:
            out[col] = out[col].fillna("")
    return out

def load_p123_p456_appended(src: Path | None = None) -> pd.DataFrame:
    p123_enr = load_p123_with_ywt_group_enriched(src)
    p456_enr = load_p456_with_group_enriched(src)   # use group-enriched P456
    # align columns
    cols = sorted(set(p123_enr.columns).union(p456_enr.columns))
    p123_enr = p123_enr.reindex(columns=cols)
    p456_enr = p456_enr.reindex(columns=cols)
    appended = pd.concat([p123_enr, p456_enr], ignore_index=True)
    return appended

# --------- CLI preview ---------
def main():
    errors = []

    # P123 + wait time
    try:
        p123_ywt = load_p123_with_ywt()
        print("[i] P123+YWT:", p123_ywt.shape)
    except Exception as e:
        errors.append(f"P123+YWT: {e!r}")

    # P456
    try:
        p456 = load_p456_enriched()
        print("[i] P456 enriched:", p456.shape)
    except Exception as e:
        errors.append(f"P456 enriched: {e!r}")

    # Append
    try:
        appended = load_p123_p456_appended()
        print("[i] Appended P123+P456:", appended.shape)
        print(appended.head(5).to_string(index=False))
    except Exception as e:
        errors.append(f"Appended: {e!r}")

    # Group
    try:
        grp = load_group_filtered()
        print(f"[i] Group filtered: {grp.shape}")
    except Exception as e:
        errors.append(f"Group: {e!r}")

    # YMS
    try:
        yms = load_yms_filtered()
        print(f"[i] YMS filtered: {yms.shape}")
    except Exception as e:
        errors.append(f"YMS: {e!r}")

    # SS
    try:
        ss = load_ss_enriched()
        print(f"[i] SS enriched: {ss.shape}")
    except Exception as e:
        errors.append(f"SS: {e!r}")

    # Month lookup
    try:
        month_lu = build_month_lookup()
        print(f"[i] Month lookup rows: {len(month_lu)}")
    except Exception as e:
        errors.append(f"Month lookup: {e!r}")

    if errors:
        print("\n[!] Loader notes:")
        for msg in errors:
            print("   -", msg)

    return locals().get("appended", locals().get("p123_ywt", None))

# ------------------ Public API ------------------
__all__ = [
    "build_month_lookup",
    "add_month_columns",
    "unpivot_p123_setup",
    "load_p123_setup_long",
    "load_wait_time",
    "load_p123_with_ywt",
    "load_group_filtered",
    "load_yms_filtered",
    "load_ss_enriched",
    "load_p456_enriched",
    "load_p456_with_group_enriched",
    "load_p123_with_ywt_group_enriched",
    "load_p123_p456_appended",
    "get_tables",
]

def get_tables(src: Path | None = None, show_errors: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load and return all key tables as DataFrames, keyed by short names.

    Returns keys (when available):
      p123_long, p123_ywt, p123_enriched, p456_enriched, p456_group_enriched,
      appended, group, yms, ss, month_lookup
    """
    out: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    def _try_put(key: str, fn):
        try:
            if key == "month_lookup":
                out[key] = fn()
            else:
                out[key] = fn(src)
        except Exception as e:
            if show_errors:
                errors.append(f"{key}: {e!r}")

    _try_put("month_lookup", build_month_lookup)
    _try_put("p123_long", load_p123_setup_long)
    _try_put("p123_ywt",  load_p123_with_ywt)
    _try_put("p123_enriched", load_p123_with_ywt_group_enriched)
    _try_put("p456_enriched", load_p456_enriched)
    _try_put("p456_group_enriched", load_p456_with_group_enriched)  # new
    _try_put("appended", load_p123_p456_appended)
    _try_put("group", load_group_filtered)
    _try_put("yms",   load_yms_filtered)
    _try_put("ss",    load_ss_enriched)

    if show_errors and errors:
        out["__errors__"] = pd.DataFrame({"error": errors})
    return out

def _cli_preview():
    tables = get_tables(show_errors=True)
    for k, v in tables.items():
        try:
            shape = v.shape  # DataFrame
        except Exception:
            shape = "<n/a>"
        print(f"[i] {k}: {shape}")
    if "__errors__" in tables:
        print("\n[!] Loader notes:")
        for msg in tables["__errors__"]["error"].tolist():
            print("   -", msg)

if __name__ == "__main__":
    _cli_preview()
