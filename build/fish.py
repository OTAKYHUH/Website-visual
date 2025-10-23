# build_fish.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import tempfile, shutil, re, calendar
from typing import Optional, Dict

# ========= CONFIG =========
TARGET_ROOT = Path(__file__).resolve().parents[1] / "static"
GLOB_PATTERN = "**/*.xlsx"
TARGET_SHEET = "fish"  # case-insensitive

# ========= tiny file helpers (same vibe as your other build_*.py) =========
def _pick_latest_excel(root: Path, pattern: str = GLOB_PATTERN) -> Optional[Path]:
    cands = [p for p in root.glob(pattern) if p.is_file() and not p.name.startswith("~$")]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

def _temp_copy(path: Path) -> Path:
    td = Path(tempfile.mkdtemp(prefix="xlsx_"))
    dst = td / path.name
    shutil.copy2(path, dst)
    return dst

def _has_sheet(xlsx: Path, name_ci: str) -> bool:
    try:
        with pd.ExcelFile(xlsx) as xf:
            return any(str(s).strip().lower() == name_ci.strip().lower() for s in xf.sheet_names)
    except Exception:
        return False

def _prefer_master(root: Path) -> Optional[Path]:
    for n in ("All.xlsx", "all.xlsx"):
        p = root / n
        if p.is_file():
            return p
    alls = [p for p in root.rglob("All*.xlsx") if p.is_file() and not p.name.startswith("~$")]
    return max(alls, key=lambda p: p.stat().st_mtime) if alls else None

def _resolve_source(src: Path | None = None) -> Path:
    """
    1) If src is a file, use it.
    2) If src is a dir or None, prefer All.xlsx (if it has the sheet); else newest .xlsx that has the sheet.
    3) Last resort: newest .xlsx (even if we can't pre-check the sheet).
    """
    if src:
        src = Path(src)
        if src.is_file():
            return src
        if not src.is_dir():
            raise FileNotFoundError(f"Not found: {src}")
        root = src
    else:
        root = TARGET_ROOT if TARGET_ROOT.exists() else Path.cwd()

    master = _prefer_master(root)
    if master and _has_sheet(master, TARGET_SHEET):
        return master

    newest = None
    newest_mtime = -1
    for p in root.rglob("*.xlsx"):
        if p.name.startswith("~$"):
            continue
        if _has_sheet(p, TARGET_SHEET) and p.stat().st_mtime > newest_mtime:
            newest, newest_mtime = p, p.stat().st_mtime
    if newest:
        return newest

    latest = _pick_latest_excel(root)
    if latest:
        return latest

    raise FileNotFoundError("Could not resolve a source Excel for FISH.")

def _read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame:
    tmp = _temp_copy(xlsx)
    try:
        with pd.ExcelFile(tmp) as xf:
            # case-insensitive
            sname = None
            for s in xf.sheet_names:
                if str(s).strip().lower() == sheet.strip().lower():
                    sname = s; break
            if sname is None:
                # fallback: partial match (e.g., "Fish Records")
                for s in xf.sheet_names:
                    if sheet.strip().lower() in str(s).strip().lower():
                        sname = s; break
            if sname is None:
                raise KeyError(f"Sheet '{sheet}' not found in {xlsx.name}")
            return pd.read_excel(xf, sheet_name=sname)
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass

# ========= utils =========
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clean = []
    for c in df.columns:
        s = str(c).replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        clean.append(s)
    df.columns = clean
    return df

def _clean_text(s: pd.Series) -> pd.Series:
    out = (s.astype(str)
             .str.replace("\u00A0", " ", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())
    out = out.mask(out.str.lower().isin(["nan", "none", "-", "--"]), "")
    return out

def _month_to_num_and_abbr(month_val) -> tuple[pd.Series, pd.Series]:
    """
    Accepts:
      - 'Jan' / 'January'
      - 1 / '1' / '01'
      - dates (we'll extract month)
    Returns two pd.Series: (MonthNum, MonthAbbr 'Jan')
    """
    s = pd.Series(month_val)
    # Try datetime first (covers real dates or strings with dates)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False, cache=True)
    # Where datetime parsed, use dt.month
    monthnum = dt.dt.month

    # For non-parsed rows, handle strings / ints
    mask_unparsed = monthnum.isna()
    if mask_unparsed.any():
        raw = s[mask_unparsed].astype(str).str.strip()

        # If numeric-like (e.g., '1', '01', '12')
        num_like = pd.to_numeric(raw, errors="coerce")
        m_from_num = num_like.where(~num_like.isna() & (num_like.between(1,12)))

        # If text like 'Jan' or 'January'
        abbr_map = {calendar.month_abbr[i].lower(): i for i in range(1,13)}
        full_map = {calendar.month_name[i].lower(): i for i in range(1,13)}
        txt = raw.str.lower()
        m_from_txt = txt.map(abbr_map).fillna(txt.map(full_map))

        # combine numeric then text
        m_combined = m_from_num.fillna(m_from_txt)
        monthnum = monthnum.fillna(m_combined)

    monthnum = monthnum.astype("Int64")
    monthabbr = monthnum.map({i: calendar.month_abbr[i] for i in range(1,13)})
    return monthnum, monthabbr

def build_month_lookup() -> pd.DataFrame:
    rows = [{"MonthNum": i,
             "Month": calendar.month_name[i],
             "MonthAbbr": calendar.month_abbr[i],
             "MonthKey": f"{i:02d}"} for i in range(1, 13)]
    return pd.DataFrame(rows)

# ========= core cleaners/loaders =========
def _clean_fish_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: Name, Month, Fish Score  (case-insensitive / flexible).
    Standardize to: Name, Month, MonthNum, FishScore
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Name","Month","MonthNum","FishScore"])

    df = _normalize_columns(df)

    # find columns (case-insensitive match by canonical names)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    def pick(*options):
        for o in options:
            if o.lower() in cols_lower:
                return cols_lower[o.lower()]
        return None

    name_col = pick("Name", "Employee", "Staff", "Person")
    month_col = pick("Month")
    score_col = pick("Fish Score", "FishScore", "Score", "Points")

    if not name_col or not month_col or not score_col:
        raise KeyError("Fish sheet must contain columns for Name, Month, and Fish Score.")

    out = df[[name_col, month_col, score_col]].copy()

    # standardize
    out = out.rename(columns={
        name_col: "Name",
        month_col: "Month",
        score_col: "FishScore",
    })
    out["Name"] = _clean_text(out["Name"])

    # MonthNum + Month (Abbr)
    monthnum, monthabbr = _month_to_num_and_abbr(out["Month"])
    out["MonthNum"] = monthnum
    out["Month"] = monthabbr  # enforce 'Jan' style

    # numeric score
    out["FishScore"] = pd.to_numeric(out["FishScore"], errors="coerce").fillna(0)

    # sort nice
    out = out.sort_values(["Name","MonthNum"], kind="stable").reset_index(drop=True)
    return out

def load_fish(src: Path | None = None) -> pd.DataFrame:
    xlsx = _resolve_source(src)
    raw = _read_sheet(xlsx, TARGET_SHEET)
    return _clean_fish_base(raw)

def fish_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated monthly fish scores per person.
    If the base is already monthly, this is effectively an identity/groupby.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Name","Month","MonthNum","FishScore"])
    g = (df.groupby(["Name","Month","MonthNum"], dropna=False)["FishScore"]
           .sum()
           .reset_index())
    return g.sort_values(["Name","MonthNum"], kind="stable").reset_index(drop=True)

# ========= Public API =========
__all__ = [
    "build_month_lookup",
    "load_fish",
    "fish_monthly",
    "get_tables",
]

def get_tables(src: Path | None = None, show_errors: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    errs: list[str] = []
    try:
        base = load_fish(src)
        out["fish"] = base
        out["fish_monthly"] = fish_monthly(base)
        out["month_lookup"] = build_month_lookup()
    except Exception as e:
        if show_errors:
            errs.append(f"fish: {e!r}")
    if show_errors and errs:
        out["__errors__"] = pd.DataFrame({"error": errs})
    return out

# ========= CLI preview =========
def _cli_preview():
    tables = get_tables(show_errors=True)
    for k, v in tables.items():
        shape = getattr(v, "shape", "<n/a>")
        print(f"[i] {k}: {shape}")
    if "__errors__" in tables:
        print("\n[!] Loader notes:")
        for msg in tables["__errors__"]["error"].tolist():
            print("   -", msg)

if __name__ == "__main__":
    _cli_preview()
