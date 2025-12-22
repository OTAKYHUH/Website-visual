# build_attendance.py — merge multiple sheets then tally from monthly columns only (FAST CACHED VERSION)
from __future__ import annotations

import os
import re
import glob
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd

# ---------- config ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS: List[Path] = [ROOT / "data", ROOT / "static", ROOT]
PATTERNS  = ["*Attendance*.xlsx", "*Attendance*.xlsm", "*MC*.xlsx", "*MC*.xlsm", "*.xlsx", "*.xlsm"]
MONTHS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_BASES = ("MC","UL","TURNOUT")

# Preferred sheet names to merge (case-insensitive, allow spacing variants)
PREFERRED_SHEETS = [
    r"^P1[-\s]?3\s*\(YOU\)$",
    r"^P1[-\s]?3\s*\(YOC\)$",
    r"^P4[-\s]?6$",
]

# ---------- performance knobs ----------
# Cache merged workbook + lookup maps (THIS is the big speed-up)
_ATT_CACHE_TTL = 600  # seconds
_FILELIST_TTL = 120   # seconds (folder scanning cache)

# Log verbosity:
#   ATT_VERBOSE=1 -> prints [ATT DATA] per person (like you see now)
#   ATT_VERBOSE=0 -> prints only on cache refresh (recommended)
#ATT_VERBOSE = os.environ.get("ATT_VERBOSE", "1").strip() not in {"0", "false", "False", "no", "NO"}
ATT_VERBOSE = False

# Quiet the openpyxl warning spam (optional, but keeps console clean)
warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
)

# ---------- caches ----------
_FILELIST_CACHE: Dict[str, Any] = {"ts": 0.0, "key": None, "files": []}

# Holds the best merged DF + lookup maps for fast row matching
_ATT_MASTER_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "key": None,         # cache key ("__AUTO__" or explicit path)
    "picked": None,      # Path
    "picked_sig": None,  # (path, mtime_ns, size)
    "df": None,          # merged DataFrame
    "used_sheets": [],

    # lookup meta
    "month_cols": [],
    "score": None,       # Series[int] per row

    "col_emp": None,
    "col_stf": None,
    "col_nam": None,

    # maps: normalized id/name -> row index with max score
    "map_emp": {},
    "map_stf": {},
    "map_namekey": {},
    "map_nospace": {},
    "map_ini": {},       # initials -> best row index
}


# ---------- helpers ----------
def _norm_col(c: object) -> str:
    s = "" if c is None else str(c)
    s = re.sub(r"\s+", "_", s.strip())
    return s.upper()


def _name_key(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _name_nospace(s) -> str:
    t = "" if pd.isna(s) else str(s)
    return re.sub(r"[^A-Z0-9]+", "", t.upper())


def _initials_from_namekey(nk: str) -> str:
    # nk is already uppercase words separated by spaces
    parts = [p for p in nk.split() if p and p not in {"D", "O"}]
    return "".join(p[0] for p in parts if p)


def _to_int(v) -> int:
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def _cache_key(path: Optional[str]) -> str:
    if path:
        return str(Path(path).resolve())
    return "__AUTO__"


def _file_sig(p: Path) -> tuple[str, int, int]:
    st = p.stat()
    return (str(p), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))


def _find_files(explicit: Optional[str]) -> List[Path]:
    ck = _cache_key(explicit)
    now = time.time()

    # If explicit, don't cache-list; return direct file.
    if explicit:
        p = Path(explicit)
        return [p] if p.exists() else []

    # Cached folder scan
    if _FILELIST_CACHE.get("key") == ck and (now - float(_FILELIST_CACHE.get("ts", 0.0))) <= _FILELIST_TTL:
        return list(_FILELIST_CACHE.get("files") or [])

    files: List[Path] = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for pat in PATTERNS:
            files.extend(sorted(d.glob(pat)))

    if not files:
        for d in DATA_DIRS:
            if not d.exists():
                continue
            hits = glob.glob(str(d / "**" / "*.xlsx"), recursive=True) + \
                   glob.glob(str(d / "**" / "*.xlsm"), recursive=True)
            files.extend(Path(h) for h in hits)

    seen, uniq = set(), []
    for f in files:
        if f not in seen and f.exists():
            uniq.append(f)
            seen.add(f)

    _FILELIST_CACHE.update({"ts": now, "key": ck, "files": uniq})
    return uniq


def _row_has_month_cols(cols_norm: List[str]) -> bool:
    s = set(cols_norm)
    found = 0
    for m in MONTHS:
        mN = _norm_col(m)
        for base in MONTH_BASES:
            if f"{mN}_{base}" in s or f"{mN}{base}" in s:
                found += 1
                if found >= 3:
                    return True
    return False


def _df_has_month_cols(df: pd.DataFrame) -> bool:
    return _row_has_month_cols(list(df.columns))


def _sheet_month_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    s = set(df.columns)
    for m in MONTHS:
        mN = _norm_col(m)
        for base in MONTH_BASES:
            a, b = f"{mN}_{base}", f"{mN}{base}"
            if a in s:
                out.append(a)
            elif b in s:
                out.append(b)
            else:
                for c in s:
                    if str(c).startswith(a + "_") or str(c).startswith(b + "_"):
                        out.append(c)
                        break
    return out


def _parse_sheet(xls: pd.ExcelFile, sh: str) -> Optional[pd.DataFrame]:
    try:
        raw = xls.parse(sh, header=None, dtype=object)
    except Exception:
        return None

    # detect header row
    header_idx: Optional[int] = None
    for i in range(min(len(raw), 40)):
        cols_test = [_norm_col(v) for v in raw.iloc[i].tolist()]
        if _row_has_month_cols(cols_test):
            header_idx = i
            break
    if header_idx is None:
        for i in range(min(len(raw), 10)):
            if raw.iloc[i].notna().any():
                header_idx = i
                break
        if header_idx is None:
            header_idx = 0

    cols_src = raw.iloc[header_idx].tolist() if header_idx < len(raw) else []
    cols_norm = [_norm_col(c) for c in cols_src]

    # uniquify headers
    seen = {}
    final_cols = []
    for c in cols_norm:
        if not c:
            c = "COL"
        seen[c] = seen.get(c, 0) + 1
        final_cols.append(f"{c}_{seen[c]}" if seen[c] > 1 else c)

    df = raw.iloc[min(header_idx + 1, len(raw)) :].copy()
    df.columns = final_cols if final_cols else []
    return df.reset_index(drop=True)


def _load_and_merge_file(xl: Path) -> tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load a workbook and concatenate:
      - First, any sheets that match PREFERRED_SHEETS
      - Then, any other sheets that have month columns (as a supplement)

    NOTE: This is expensive — we cache the chosen result so it runs once per TTL.
    """
    merged: List[pd.DataFrame] = []
    used_names: List[str] = []

    try:
        # read_only speeds openpyxl for large files; data_only avoids formula evaluation
        xls = pd.ExcelFile(xl, engine="openpyxl", engine_kwargs={"data_only": True, "read_only": True})
    except Exception:
        try:
            xls = pd.ExcelFile(xl)
        except Exception:
            return None, used_names

    names = xls.sheet_names

    # 1) preferred sheets
    picked_idx = set()
    for pat in PREFERRED_SHEETS:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for i, nm in enumerate(names):
            if i in picked_idx:
                continue
            if rx.search(nm or ""):
                df = _parse_sheet(xls, nm)
                if df is not None and _df_has_month_cols(df):
                    merged.append(df)
                    used_names.append(nm)
                    picked_idx.add(i)

    # 2) supplement with any other sheet that has month columns
    for i, nm in enumerate(names):
        if i in picked_idx:
            continue
        df = _parse_sheet(xls, nm)
        if df is not None and _df_has_month_cols(df):
            merged.append(df)
            used_names.append(nm)
            picked_idx.add(i)

    if not merged:
        return None, used_names

    combo = pd.concat(merged, ignore_index=True, sort=False)
    return combo, used_names


def _pick_best_file(files: List[Path]) -> tuple[Optional[pd.DataFrame], Optional[Path], List[str]]:
    """
    For each file, merge its candidate sheets, score by:
      (number of month columns, sum of all month columns), pick the best file.
    """
    best_df, best_file, best_sheets, best_score = None, None, [], (-1, -1.0)

    for f in files:
        df, used = _load_and_merge_file(f)
        if df is None or df.empty:
            continue
        cols = _sheet_month_columns(df)
        if not cols:
            continue
        try:
            total_sum = 0.0
            # sum without building an intermediate DataFrame (less memory)
            for c in cols:
                total_sum += float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        except Exception:
            total_sum = 0.0
        score = (len(cols), total_sum)
        if score > best_score:
            best_df, best_file, best_sheets, best_score = df, f, used, score

    return best_df, best_file, best_sheets


def _col(df: pd.DataFrame, *opts: str) -> Optional[str]:
    have = set(df.columns)
    for o in opts:
        k = _norm_col(o)
        if k in have:
            return k
    for o in opts:
        k = _norm_col(o)
        for c in have:
            if c == k or str(c).startswith(k + "_"):
                return c
    return None


def _build_lookup_cache(df: pd.DataFrame) -> None:
    """
    Precompute:
      - month_cols
      - per-row score (sum of month columns)
      - best row index per key for staff_no / employee_id / namekey / nospace / initials
    """
    month_cols = _sheet_month_columns(df)

    # compute score series once (sum of all month columns)
    score = pd.Series(0.0, index=df.index)
    for c in month_cols:
        score = score + pd.to_numeric(df.get(c), errors="coerce").fillna(0)
    score = score.fillna(0).astype(int)

    col_emp = _col(df, "Employee ID", "EMPLOYEE ID", "Emp ID", "EMP_ID")
    col_stf = _col(df, "Staff No", "STAFF NO", "StaffNo", "STAFF_NO")
    col_nam = _col(df, "Staff Name (JO)", "Staff Name", "Name", "NAME")

    map_emp: Dict[str, int] = {}
    map_stf: Dict[str, int] = {}
    map_namekey: Dict[str, int] = {}
    map_nospace: Dict[str, int] = {}
    map_ini: Dict[str, int] = {}

    # helper: idxmax per group key
    def _idxmax_map(keys: pd.Series) -> Dict[str, int]:
        tmp = pd.DataFrame({"k": keys, "s": score})
        tmp["k"] = tmp["k"].astype(str)
        tmp = tmp[tmp["k"].notna()]
        tmp["k"] = tmp["k"].str.strip()
        tmp = tmp[tmp["k"] != ""]
        if tmp.empty:
            return {}
        idx = tmp.groupby("k")["s"].idxmax()  # returns original index labels
        out = {k: int(i) for k, i in idx.items()}
        return out

    try:
        if col_stf:
            k_stf = df[col_stf].astype(str).str.upper().str.strip()
            map_stf = _idxmax_map(k_stf)

        if col_emp:
            k_emp = df[col_emp].astype(str).str.upper().str.strip()
            map_emp = _idxmax_map(k_emp)

        if col_nam:
            # namekey / nospace / initials computed once
            names = df[col_nam].astype(str)

            k_namekey = names.map(_name_key)
            map_namekey = _idxmax_map(k_namekey)

            k_nospace = names.map(_name_nospace)
            map_nospace = _idxmax_map(k_nospace)

            k_ini = k_namekey.map(_initials_from_namekey)
            map_ini = _idxmax_map(k_ini)
    except Exception:
        pass

    _ATT_MASTER_CACHE.update({
        "month_cols": month_cols,
        "score": score,
        "col_emp": col_emp,
        "col_stf": col_stf,
        "col_nam": col_nam,
        "map_emp": map_emp,
        "map_stf": map_stf,
        "map_namekey": map_namekey,
        "map_nospace": map_nospace,
        "map_ini": map_ini,
    })


def _get_best_merged_df(path: Optional[str]) -> tuple[Optional[pd.DataFrame], Optional[Path], List[str]]:
    """
    Return (df, picked_file, used_sheets), using cache to avoid re-opening Excel each call.
    """
    now = time.time()
    ck = _cache_key(path)

    hit = (_ATT_MASTER_CACHE.get("key") == ck) and (_ATT_MASTER_CACHE.get("df") is not None)
    fresh = (now - float(_ATT_MASTER_CACHE.get("ts", 0.0))) <= _ATT_CACHE_TTL

    if hit and fresh:
        # If we have a picked file, also ensure it hasn't changed
        picked: Optional[Path] = _ATT_MASTER_CACHE.get("picked")
        sig = _ATT_MASTER_CACHE.get("picked_sig")
        if picked and picked.exists() and sig:
            try:
                if _file_sig(picked) == sig:
                    return _ATT_MASTER_CACHE["df"], picked, list(_ATT_MASTER_CACHE.get("used_sheets") or [])
            except Exception:
                pass
        else:
            # no file sig to validate; still return cached
            return _ATT_MASTER_CACHE["df"], picked, list(_ATT_MASTER_CACHE.get("used_sheets") or [])

    # refresh: scan + pick best
    files = _find_files(path)
    df, picked, used_sheets = _pick_best_file(files)

    if df is None or df.empty or picked is None:
        # cache negative briefly to avoid hammering disk
        _ATT_MASTER_CACHE.update({
            "ts": now,
            "key": ck,
            "picked": None,
            "picked_sig": None,
            "df": None,
            "used_sheets": [],
            "month_cols": [],
            "score": None,
            "map_emp": {},
            "map_stf": {},
            "map_namekey": {},
            "map_nospace": {},
            "map_ini": {},
        })
        return None, None, []

    _ATT_MASTER_CACHE.update({
        "ts": now,
        "key": ck,
        "picked": picked,
        "picked_sig": _file_sig(picked),
        "df": df,
        "used_sheets": used_sheets,
    })
    _build_lookup_cache(df)

    if ATT_VERBOSE:
        sheet_info = ",".join(used_sheets) if used_sheets else "?"
        print(f"[ATT CACHE REFRESH] picked={picked.name} sheets=[{sheet_info}] rows={len(df)}")

    return df, picked, used_sheets


# ---------- matching & extraction ----------
def _match_row_fast(
    df: pd.DataFrame,
    name: Optional[str],
    employee_id: Optional[str],
    staff_no: Optional[str],
) -> Optional[pd.Series]:
    """
    FAST: uses precomputed maps from _build_lookup_cache()
    """
    if df is None or df.empty:
        return None

    # 1) staff no (exact)
    if staff_no:
        k = str(staff_no).upper().strip()
        idx = _ATT_MASTER_CACHE.get("map_stf", {}).get(k)
        if idx is not None:
            return df.loc[idx]

    # 2) employee id (exact)
    if employee_id:
        k = str(employee_id).upper().strip()
        idx = _ATT_MASTER_CACHE.get("map_emp", {}).get(k)
        if idx is not None:
            return df.loc[idx]

    if not name:
        return None

    # 3) exact namekey
    nk = _name_key(name)
    idx = _ATT_MASTER_CACHE.get("map_namekey", {}).get(nk)
    if idx is not None:
        return df.loc[idx]

    # 4) nospace
    ns = _name_nospace(name)
    idx = _ATT_MASTER_CACHE.get("map_nospace", {}).get(ns)
    if idx is not None:
        return df.loc[idx]

    # 5) initials prefix (fallback)
    ini = _initials_from_namekey(nk)
    if ini:
        ini_map: Dict[str, int] = _ATT_MASTER_CACHE.get("map_ini", {}) or {}
        # find any initials that startwith ini, pick best by score
        best_idx = None
        best_score = -1
        score: Optional[pd.Series] = _ATT_MASTER_CACHE.get("score")
        for full_ini, ridx in ini_map.items():
            if full_ini.startswith(ini):
                sc = int(score.loc[ridx]) if isinstance(score, pd.Series) else 0
                if sc > best_score:
                    best_score, best_idx = sc, ridx
        if best_idx is not None:
            return df.loc[best_idx]

    return None


def _extract_monthly(row: pd.Series) -> Dict[str, Dict[str, int]]:
    monthly = {m: {"mc": 0, "ul": 0, "turnout": 0} for m in MONTHS}
    idx = set(row.index)

    def pick(m: str, base: str) -> Optional[str]:
        m_norm = _norm_col(m)
        for shape in (f"{m_norm}_{base}", f"{m_norm}{base}"):
            if shape in idx:
                return shape
            for c in idx:
                if str(c).startswith(shape + "_"):
                    return c
        return None

    for m in MONTHS:
        mc = pick(m, "MC")
        ul = pick(m, "UL")
        to = pick(m, "TURNOUT")
        if mc:
            monthly[m]["mc"] = _to_int(row[mc])
        if ul:
            monthly[m]["ul"] = _to_int(row[ul])
        if to:
            monthly[m]["turnout"] = _to_int(row[to])

    return monthly


def _latest_nonzero_month(monthly: Dict[str, Dict[str, int]]) -> str:
    for m in reversed(MONTHS):
        if any(monthly[m].get(k, 0) for k in ("mc", "ul", "turnout")):
            return m
    return pd.Timestamp.now().strftime("%b")


def _stats(row: Optional[pd.Series], req_month: Optional[str]) -> Dict[str, object]:
    if row is None or (isinstance(row, pd.Series) and row.empty):
        m = (req_month or pd.Timestamp.now().strftime("%b"))[:3].title()
        out = {
            "month": m,
            "month_mc": 0, "month_ul": 0, "month_turnout": 0,
            "total_mc": 0,  "total_ul": 0,  "total_turnout": 0,
            "monthly": {mn: {"mc": 0, "ul": 0, "turnout": 0} for mn in MONTHS},
            "not_found": True,
        }
        if ATT_VERBOSE:
            print(f"[ATT DATA] <none> | req_month {m} | got_total 0 0 0 | got_month {m} | not_found True")
        return out

    monthly = _extract_monthly(row)
    tm = sum(int(monthly[m]["mc"]) for m in MONTHS)
    tu = sum(int(monthly[m]["ul"]) for m in MONTHS)
    tt = sum(int(monthly[m]["turnout"]) for m in MONTHS)

    if req_month:
        ask = req_month.strip()[:3].title()
        month_used = "Jun" if ask == "Jun" else ask
    else:
        month_used = _latest_nonzero_month(monthly)

    mm = monthly.get(month_used, {"mc": 0, "ul": 0, "turnout": 0})

    out = {
        "month": month_used,
        "month_mc": int(mm.get("mc", 0)),
        "month_ul": int(mm.get("ul", 0)),
        "month_turnout": int(mm.get("turnout", 0)),
        "total_mc": int(tm),
        "total_ul": int(tu),
        "total_turnout": int(tt),
        "monthly": monthly,
        "not_found": False,
    }

    if ATT_VERBOSE:
        who = _name_key(row.get("STAFF_NAME_(JO)") or row.get("STAFF_NAME") or row.get("NAME") or "")
        print(
            f"[ATT DATA] {who or '<row>'} | req_month {req_month or 'auto'} | "
            f"got_total {out['total_mc']} {out['total_ul']} {out['total_turnout']} | got_month {out['month']}"
        )
    return out


def _fallback_from_mc(
    month: Optional[str],
    name: Optional[str],
    employee_id: Optional[str],
    staff_no: Optional[str],
) -> Dict[str, object]:
    try:
        from .build_MC import get_tables as mc_get  # type: ignore
        mc_payload = mc_get(month=month, name=name, employee_id=employee_id, staff_no=staff_no) or {}
        mc = mc_payload.get("mc_details", {}) or {}
        m = (month or pd.Timestamp.now().strftime("%b"))[:3].title()
        at = {
            "month": m,
            "month_mc": int(mc.get("month_mc", 0)),
            "month_ul": int(mc.get("month_ul", 0)),
            "month_turnout": int(mc.get("month_turnout", 0)),
            "total_mc": int(mc.get("total_mc", 0)),
            "total_ul": int(mc.get("total_ul", 0)),
            "total_turnout": int(mc.get("total_turnout", 0)),
            "monthly": mc.get("monthly", {mn: {"mc": 0, "ul": 0, "turnout": 0} for mn in MONTHS}),
            "not_found": bool(mc.get("not_found", False)),
        }
        if ATT_VERBOSE:
            print(
                f"[ATT DATA] (fallback MC) | month {at['month']} | "
                f"got_total {at['total_mc']} {at['total_ul']} {at['total_turnout']}"
            )
        return at
    except Exception:
        return _stats(None, month)


# ---------- public API ----------
def get_tables(
    month: Optional[str] = None,
    name: Optional[str] = None,
    employee_id: Optional[str] = None,
    staff_no: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Same API as your original. Big changes:
      - caches workbook selection + merged df
      - caches lookup maps so matching per person is O(1)
    """
    try:
        df, picked, used_sheets = _get_best_merged_df(path)

        if df is None or df.empty:
            return {"attendance_stats": _fallback_from_mc(month, name, employee_id, staff_no)}

        row = _match_row_fast(df, name, employee_id, staff_no)
        stats = _stats(row, month)

        # Print [ATT PICK] only when cache refresh OR when verbose mode is on
        if ATT_VERBOSE:
            try:
                sheet_info = ",".join(used_sheets) if used_sheets else "?"
                print(
                    f"[ATT PICK] file={picked.name if picked else '?'} "
                    f"sheets=[{sheet_info}] totals={stats['total_mc']}/{stats['total_ul']}/{stats['total_turnout']}"
                )
            except Exception:
                pass

        return {"attendance_stats": stats}

    except Exception as e:
        if ATT_VERBOSE:
            print(f"[ATT ERROR] {type(e).__name__} {e}")
        return {"attendance_stats": _fallback_from_mc(month, name, employee_id, staff_no)}
