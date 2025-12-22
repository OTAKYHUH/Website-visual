# routes/main.py
from __future__ import annotations

import os, re, time, calendar
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Blueprint, render_template, request

# ---- external builders (still needed) ----
from build.build_data import get_tables as get_main_tables  # type: ignore
from build.build_safety import get_tables as get_safety_tables  # type: ignore

try:
    from build.photos import get_tables as get_photo_tables  # type: ignore
except Exception:
    get_photo_tables = None

# HSL builders
from build.build_H import get_tables as get_hsl_tables  # type: ignore
try:
    # optional fast path if you added it in build_H.py
    from build.build_H import get_log_only as get_hsl_log_only  # type: ignore
except Exception:
    get_hsl_log_only = None

# Attendance (same builder used by profile.py / nonshift.py)
try:
    from build.build_attendance import get_tables as get_attendance_tables  # type: ignore
    print("[ATT LOADED]", getattr(get_attendance_tables, "__version__", "?"))
except Exception:
    get_attendance_tables = None
    print("[ATT LOADED] FAILED")

main = Blueprint("main", __name__, template_folder="templates")

# ---------- caches ----------
_CACHE_TTL = 24 * 3600
_MAIN_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}

_PHOTO_TTL = 24 * 3600
_PHOTO_CACHE: Dict[str, object] = {"ts": 0.0, "photos": {}}

_PEOPLE_TTL = 24 * 3600
_PEOPLE_CACHE: Dict[Tuple[Tuple[str, ...], str], Dict[str, object]] = {}

_MONTHS_TTL = 24 * 3600
_MONTHS_CACHE: Dict[str, object] = {"ts": 0.0, "months": None}

_SAFETY_TTL = 24 * 3600
_SAFETY_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}

_NAME_MAP_TTL = 24 * 3600
_NAME_MAP_CACHE: Dict[Tuple, dict] = {}

_ATT_PERSON_TTL = 24 * 3600
_ATT_PERSON_CACHE: Dict[tuple, dict] = {}

# <<< PERF >>> heavy-work caches (safe; no logic changes)
_BASE_CACHE: Dict[str, object] = {"ts": 0.0, "df": None}          # base concat+month only
_HSL_TTL = 24 * 3600
_HSL_CACHE: Dict[str, object] = {"ts": 0.0, "rates_df": None}     # parsed HSL log rates

PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"
MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

_RAHIMI_RX = r"\brahimi\b"  # no match-groups warning, faster

# ---------- tiny utils ----------
def _now() -> float:
    return time.time()


# <<< PERF >>> cache wrappers (NO logic changes)
def _get_cached_base(builder):
    now = _now()
    if _BASE_CACHE["df"] is not None and (now - float(_BASE_CACHE["ts"])) <= _CACHE_TTL:
        return _BASE_CACHE["df"]
    df = builder()
    _BASE_CACHE.update({"ts": now, "df": df})
    return df


def _get_cached_hsl():
    now = _now()
    hit = _HSL_CACHE.get("rates_df")
    if isinstance(hit, pd.DataFrame) and (now - float(_HSL_CACHE.get("ts", 0.0))) <= _HSL_TTL:
        return hit

    df = _extract_hsl_rates_from_log()
    _HSL_CACHE["rates_df"] = df
    _HSL_CACHE["ts"] = now
    return df


def _get_cached_name_map(cache_key, safety_names, base_names):
    now = _now()
    hit = _NAME_MAP_CACHE.get(cache_key)
    if hit and (now - float(hit.get("ts", 0.0))) <= _NAME_MAP_TTL:
        return hit.get("map", {})  # type: ignore[return-value]
    m = _build_name_map(safety_names, base_names, threshold=0.5)
    _NAME_MAP_CACHE[cache_key] = {"ts": now, "map": m}
    return m


def _attendance_cached(exact_nm: str | None, emp_id: str | None, staff_no: str | None, att_month: str) -> dict:
    key = (
        att_month,
        str(emp_id or "").strip(),
        str(staff_no or "").strip(),
        _name_key(exact_nm or ""),
    )
    now = _now()
    hit = _ATT_PERSON_CACHE.get(key)
    if hit and (now - float(hit.get("ts", 0.0))) <= _ATT_PERSON_TTL:
        return hit.get("stats", {})

    stats = _call_attendance_for_person(exact_nm, emp_id, staff_no, att_month)
    _ATT_PERSON_CACHE[key] = {"ts": now, "stats": stats}
    return stats


def _order_months(ms) -> List[str]:
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s]


def _name_key(s) -> str:
    import re, pandas as _pd
    t = "" if _pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _display_name(n: str) -> str:
    return re.sub(r"\bD O\b", "D/O", n, flags=re.IGNORECASE) if n else n


def _norm_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    if df is None or df.empty:
        return None
    index = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_colname(cand)
        if k in index:
            return index[k]
    for alt in ("group","groupcode","empgroup","grp","terminal","plant","source"):
        if alt in index:
            return index[alt]
    return None


def _fast_parse_date(series, *, dayfirst=False):
    """Fast, tolerant date parser that avoids slow per-element dateutil fallback."""
    s = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

    st = s.astype(str).str.strip()
    candidates = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%m-%d-%Y",
        "%d %b %Y", "%d %B %Y", "%Y/%m/%d",
        "%d %b %y",  # allow 2-digit year too
    ]
    best = None
    best_hit = -1
    for fmt in candidates:
        dt = pd.to_datetime(st, format=fmt, errors="coerce")
        hit = dt.notna().sum()
        if hit > best_hit:
            best, best_hit = dt, hit
        if hit >= 0.6 * len(st):
            return dt
    if best_hit > 0:
        return best

    # final fallback (may warn if mixed)
    return pd.to_datetime(st, errors="coerce", dayfirst=dayfirst, cache=True)


def _ensure_month_3(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Month" not in df.columns:
        dc = _find_col(df, "EVENT_SHIFT_DT","DATE","Date")
        if dc:
            dt = _fast_parse_date(df[dc], dayfirst=False)
            df = df.copy()
            df["Month"] = dt.dt.strftime("%b")
        else:
            df = df.copy()
            df["Month"] = pd.NA
    else:
        df = df.copy()
        df["Month"] = df["Month"].astype(str).str[:3]
    return df


def _filter_to_abcd(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    col = df[group_col].astype(str).str.strip().str.upper()
    uses_star = col.str.match(r"^[ABCD]\*$").any()
    mask = col.isin({"A*","B*","C*","D*"}) if uses_star else col.str.match(r"^[ABCD]")
    out = df[mask]
    if out.empty and not col.empty:
        out = df[col.str.contains(r"^\s*[ABCD]\s*\*?\s*$", regex=True)]
    return out


def _which_plant(val) -> str | None:
    import pandas as _pd
    if val is None:
        return None
    try:
        if isinstance(val, float) and _pd.isna(val):
            return None
    except Exception:
        pass
    s = str(val).strip().upper().replace("/", " ")
    if not s:
        return None
    if s == "P123" or "123" in s:
        return "P123"
    if s == "P456" or "456" in s:
        return "P456"
    return None


# --- month normalization ---
def _to_month_abbr(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 12:
            return calendar.month_abbr[n]
    low = s.lower()
    full = {calendar.month_name[i].lower(): calendar.month_abbr[i] for i in range(1, 13)}
    abbr = {calendar.month_abbr[i].lower(): calendar.month_abbr[i] for i in range(1, 13)}
    if low in full:  return full[low]
    if low in abbr:  return abbr[low]
    if ("/" in s) or ("-" in s):
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return calendar.month_abbr[dt.month]
            except ValueError:
                pass
    return s[:3].title()


def _norm_keep_months(selected: list[str]) -> list[str]:
    return [m for m in (_to_month_abbr(x) for x in selected) if m]


# ---------- HSL helpers ----------
def _norm_shift_letter(s: str | None) -> str:
    s = (s or "").strip().upper()
    if s == "1":
        return "D"
    if s in {"2", "G"}:
        return "N"
    return s if s in {"D", "N"} else "D"


def _norm_df_shift_date_tuple(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    if "Shift" in g.columns:
        g["_sd_shift"] = g["Shift"].astype(str).map(_norm_shift_letter)
    else:
        g["_sd_shift"] = "D"

    date_col = next((c for c in ("Shift Date","ShiftDate","Dates","Date","START_DT","START DT","Start Date") if c in g.columns), None)
    if date_col is None:
        g["_sd_date"] = pd.NaT
        return g

    if date_col in ("Shift Date","ShiftDate"):
        # Log format is like: "02 Sep 25 D" -> we must parse "%d %b %y"
        date_only = g[date_col].astype(str).str.replace(r"\s+[DNdn]$", "", regex=True).str.strip()

        # FAST format tries (no dateutil)
        dt = pd.to_datetime(date_only, format="%d %b %y", errors="coerce")   # <-- FIX
        if dt.notna().mean() < 0.8:
            dt = dt.fillna(pd.to_datetime(date_only, format="%d %b %Y", errors="coerce"))
        if dt.notna().mean() < 0.8:
            dt = dt.fillna(pd.to_datetime(date_only, format="%d/%m/%Y", errors="coerce"))
        if dt.notna().mean() < 0.8:
            dt = dt.fillna(pd.to_datetime(date_only, format="%Y-%m-%d", errors="coerce"))
        if dt.notna().mean() < 0.8:
            dt = dt.fillna(pd.to_datetime(date_only, format="%d-%m-%Y", errors="coerce"))

        g["_sd_date"] = dt.dt.date
    else:
        g["_sd_date"] = _fast_parse_date(g[date_col], dayfirst=True).dt.date

    return g


def _pct_to_half_point(pct: float | int | None) -> float:
    """Convert a 0..100 percentage into 0.00..0.50 points (same rule as Reliability)."""
    if pct is None:
        return 0.0
    try:
        if isinstance(pct, float) and pd.isna(pct):
            return 0.0
    except Exception:
        pass
    try:
        x = float(pct)
    except Exception:
        return 0.0
    # same scaling as reliability: r% * 0.005, capped at 0.5
    return round(min(0.5, max(0.0, x) * 0.005), 2)


def _half_point_class(pts: float, *, orange_at: float = 0.43, green_at: float = 0.45) -> str:
    """Class for 0..0.5 points. Default thresholds: red <0.43, orange 0.43..0.45, green >=0.45."""
    try:
        x = float(pts)
    except Exception:
        x = 0.0
    if x >= green_at:
        return "pts-green"
    if x >= orange_at:
        return "pts-orange"
    return "pts-red"



def _extract_hsl_rates_from_log() -> pd.DataFrame:
    """Read ONLY HSL Log (fast if get_log_only exists) and turn into key=YYYY-MM-DD@Shift rows."""
    log_df = None

    # fast path (your new helper)
    if callable(get_hsl_log_only):
        try:
            log_df = get_hsl_log_only(verbose=False)
        except Exception as e:
            print("[HSL] get_hsl_log_only failed:", e)
            log_df = None

    # fallback path (works with original build_H.get_tables signature)
    if log_df is None or (isinstance(log_df, pd.DataFrame) and log_df.empty):
        try:
            bundle = get_hsl_tables() or {}
            if isinstance(bundle, dict):
                log_df = bundle.get("Log")
        except Exception as e:
            print("[HSL] fallback get_hsl_tables failed:", e)
            return pd.DataFrame(columns=["_key","pass_rate"])

    if not isinstance(log_df, pd.DataFrame) or log_df.empty:
        return pd.DataFrame(columns=["_key","pass_rate"])

    g = _norm_df_shift_date_tuple(log_df).copy()
    g = g.dropna(subset=["_sd_date","_sd_shift"])
    if g.empty:
        return pd.DataFrame(columns=["_key","pass_rate"])

    g["_key"] = pd.to_datetime(g["_sd_date"]).dt.strftime("%Y-%m-%d") + "@" + g["_sd_shift"].astype(str)

    col_good = None
    col_total = None
    for c in g.columns:
        lc = str(c).lower()
        if col_good is None and ("good" in lc or "pass" in lc):
            col_good = c
        if col_total is None and "total" in lc:
            col_total = c

    if not col_good or not col_total:
        return pd.DataFrame(columns=["_key","pass_rate"])

    g["_good"] = pd.to_numeric(g[col_good], errors="coerce").fillna(0.0)
    g["_total"] = pd.to_numeric(g[col_total], errors="coerce").fillna(0.0)

    agg = g.groupby("_key")[["_good","_total"]].sum()
    agg = agg[agg["_total"] > 0]
    if agg.empty:
        return pd.DataFrame(columns=["_key","pass_rate"])

    pct = (agg["_good"] / agg["_total"]) * 100.0
    out = pct.reset_index()
    out.columns = ["_key", "pass_rate"]
    return out


# ---------- caches fetchers ----------
def _main_tables() -> Dict[str, pd.DataFrame]:
    now = _now()
    if (not _MAIN_CACHE["tables"]) or (now - float(_MAIN_CACHE["ts"])) > _CACHE_TTL:
        _MAIN_CACHE["tables"] = get_main_tables(show_errors=True)
        _MAIN_CACHE["ts"] = now
    return _MAIN_CACHE["tables"] or {}


def _safety_tables() -> Dict[str, pd.DataFrame]:
    now = _now()
    if (not _SAFETY_CACHE["tables"]) or (now - float(_SAFETY_CACHE["ts"])) > _SAFETY_TTL:
        out: Dict[str, pd.DataFrame] = {}
        try:
            bundle = get_safety_tables()
            if isinstance(bundle, dict):
                out.update(bundle)
        except Exception as e:
            print("[SFT] build_safety.get_tables() error:", e)

        try:
            for k, v in _main_tables().items():
                if isinstance(v, pd.DataFrame) and k not in out:
                    out[k] = v
        except Exception:
            pass

        _SAFETY_CACHE["tables"] = out
        _SAFETY_CACHE["ts"] = now

    return _SAFETY_CACHE["tables"] or {}


def _photos_dict() -> Dict[str, str]:
    now = _now()
    if (now - float(_PHOTO_CACHE["ts"])) > _PHOTO_TTL or not _PHOTO_CACHE.get("photos"):
        photos: Dict[str, str] = {}
        if os.path.isdir(PHOTO_DIR) and callable(get_photo_tables):
            try:
                bundle = get_photo_tables(PHOTO_DIR) or {}
                raw = (bundle or {}).get("photos_dict", {}) or {}
                photos = {_name_key(k): v for k, v in raw.items()}
            except Exception:
                photos = {}
        _PHOTO_CACHE.update({"ts": now, "photos": photos})
    return _PHOTO_CACHE["photos"] or {}


# ---------- Designations ----------
_DESIG_FULL: Dict[str, str] = {
    "SOE": "Senior Operations Executive",
    "OE":  "Operations Executive",
    "AM":  "Assistant Manager",
}
_DESIG_RAW: Dict[str, str] = {
    "Abdul Hakim": "SOE", "Adnan": "SOE", "Austin Chue": "SOE", "Calvin": "SOE",
    "Chng ming hao": "SOE", "haidar": "OE", "Jacinta": "OE", "Joe Chan": "OE",
    "Johnny": "SOE", "Lai LiHong": "SOE", "Lim Chun Sern": "SOE", "Mervin": "SOE",
    "FAUZI": "SOE", "jamalullail": "SOE", "Farzhani": "SOE", "Jian Hong": "SOE",
    "Toh Chee Chong": "AM", "Velson": "AM", "Wan zulhelmi": "SOE",
}
_DESIG_MAP: Dict[str, str] = {_name_key(k): v for k, v in _DESIG_RAW.items()}


def _designation_for(person_name: str) -> str:
    norm = _name_key(person_name)
    best_code, best_len = None, -1
    for key, code in _DESIG_MAP.items():
        if key in norm or norm in key:
            if len(key) > best_len:
                best_code, best_len = code, len(key)
    return _DESIG_FULL.get(best_code or "", "")


def _safety_class(pts: int) -> str:
    return "pts-green" if pts >= 0.95 else "pts-red"

# ---------- helpers for Safety ----------
def _get_table_ci(bundle: dict, wanted: str) -> pd.DataFrame | None:
    wanted_k = wanted.strip().lower()
    for k, v in (bundle or {}).items():
        if isinstance(v, pd.DataFrame) and str(k).strip().lower() == wanted_k:
            return v
    for k, v in (bundle or {}).items():
        if isinstance(v, pd.DataFrame) and wanted_k in str(k).strip().lower():
            return v
    return None


def _shift_key(df, date_col: str | None, shift_col: str | None):
    """Return Series of 'YYYY-MM-DD@D|N' keys (NaN if missing pieces)."""
    import pandas as _pd
    if not date_col or not shift_col:
        return _pd.Series([_pd.NA] * len(df))

    dt = _fast_parse_date(df[date_col], dayfirst=False)
    sh = df[shift_col].astype(str).str.strip().str.upper().str[0].replace({"1":"D","2":"N","G":"N"})
    return dt.dt.strftime("%Y-%m-%d") + "@" + sh


def _find_si_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "si":
            return c
    for c in df.columns:
        if "si" in str(c).strip().lower():
            return c
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5 and (s.fillna(0) == s.fillna(0).round()).mean() > 0.9:
            return c
    return None


def _token_list(name: str) -> list[str]:
    return [t for t in _name_key(name).split() if t]


def _token_set(name: str) -> set[str]:
    return set(_token_list(name))


def _name_similarity(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    jac = inter / len(A | B)
    a_n, b_n = _name_key(a), _name_key(b)
    if a_n in b_n or b_n in a_n:
        jac += 0.4
    return jac


def _build_name_map(safety_raw: list[str], base_raw: list[str], threshold: float = 0.5) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for s in safety_raw:
        s_norm = _name_key(s)
        best_norm, best_sim = None, -1.0
        for b in base_raw:
            sim = _name_similarity(s, b)
            if sim > best_sim:
                best_norm, best_sim = _name_key(b), sim
        if best_sim >= threshold and best_norm:
            mapping[s_norm] = best_norm
    for s in safety_raw:
        s_norm = _name_key(s)
        if s_norm in mapping:
            continue
        toks = _token_list(s)[:2]
        if not toks:
            continue
        for b in base_raw:
            if set(toks).issubset(_token_set(b)):
                mapping[s_norm] = _name_key(b)
                break
    for s in safety_raw:
        s_norm = _name_key(s)
        if s_norm in mapping:
            continue
        for b in base_raw:
            bn = _name_key(b)
            if s_norm in bn or bn in s_norm:
                mapping[s_norm] = bn
                break
    return mapping


# ---------- Attendance helpers ----------
def _as_scalar(x):
    import pandas as _pd
    if x is None:
        return None
    if isinstance(x, _pd.Series):
        for v in x.tolist():
            s = ("" if _pd.isna(v) else str(v)).strip()
            if s and s.lower() != "nan":
                return s
        return None
    try:
        s = ("" if _pd.isna(x) else str(x)).strip()
    except Exception:
        try:
            s = str(x).strip()
        except Exception:
            return None
    return None if (s == "" or s.lower() == "nan") else s


def _extract_ids_map(df) -> dict[str, tuple[str|None, str|None]]:
    def _find_col(df, *cands):
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            k = c.lower()
            if k in low:
                return low[k]
        return None

    name_col = _find_col(df, "Name", "NAME")
    emp_col  = _find_col(df, "Employee ID", "EMPLOYEE ID", "Emp ID", "EMP_ID")
    stf_col  = _find_col(df, "Staff No", "STAFF NO", "StaffNo", "STAFF_NO")
    if not name_col:
        return {}

    keep = [c for c in [name_col, emp_col, stf_col] if c]
    tmp = df[keep].copy()
    tmp[name_col] = tmp[name_col].astype(str).str.strip()

    def _first_nonempty(series):
        import pandas as _pd
        for x in series:
            s = ("" if _pd.isna(x) else str(x)).strip()
            if s:
                return s
        return None

    grp = tmp.groupby(name_col, dropna=False).agg(_first_nonempty)
    out = {}
    for nm, row in grp.iterrows():
        out[str(nm).strip()] = (
            (row.get(emp_col) if emp_col else None),
            (row.get(stf_col) if stf_col else None),
        )
    return out


def _resolve_mc_ul(stats: dict, selected_months: list[str] | None) -> tuple[float|None, float|None]:
    if not isinstance(stats, dict):
        return None, None

    def to_num(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            return float(x)
        except Exception:
            try:
                return float(str(x).strip())
            except Exception:
                return None

    if selected_months:
        mon = selected_months[0][:3].title()
        mm = stats.get("month_mc"); mu = stats.get("month_ul")
        if mm is None or mu is None:
            monthly = stats.get("by_month") or stats.get("months") or stats.get("monthly") or {}
            rec = monthly.get(mon) or {}
            if mm is None: mm = (rec.get("mc") if isinstance(rec, dict) else None)
            if mu is None: mu = (rec.get("ul") if isinstance(rec, dict) else None)
        if mm is None: mm = stats.get(f"mc_{mon}") or stats.get(f"{mon}_mc")
        if mu is None: mu = stats.get(f"ul_{mon}") or stats.get(f"{mon}_ul")
        return to_num(mm), to_num(mu)

    return to_num(stats.get("total_mc")), to_num(stats.get("total_ul"))


def _call_attendance_for_person(exact_nm: str | None,
                                emp_id: str | None,
                                staff_no: str | None,
                                att_month: str) -> dict:
    if not callable(get_attendance_tables):
        return {}

    emp_id  = _as_scalar(emp_id)
    staff_no = _as_scalar(staff_no)

    if (emp_id is not None) or (staff_no is not None):
        try:
            b = get_attendance_tables(month=att_month, name=None, employee_id=emp_id, staff_no=staff_no) or {}
            stats = b.get("attendance_stats", {}) or {}
            if stats:
                return stats
        except Exception:
            pass

    nm = _as_scalar(exact_nm)
    if nm:
        try:
            b = get_attendance_tables(month=att_month, name=nm, employee_id=None, staff_no=None) or {}
            stats = b.get("attendance_stats", {}) or {}
            if stats:
                return stats
        except Exception:
            pass

    if nm:
        import re as _re
        cand = _re.sub(r"\s+", " ", nm.replace("/", " ")).strip()
        try:
            b = get_attendance_tables(month=att_month, name=cand, employee_id=None, staff_no=None) or {}
            stats = b.get("attendance_stats", {}) or {}
            if stats:
                return stats
        except Exception:
            pass

    return {}


def _id_lookup_with_fallback(id_map: dict[str, tuple[str|None, str|None]], exact_nm: str | None):
    import pandas as _pd

    def _to_pair(x):
        if isinstance(x, (tuple, list)):
            return _as_scalar(x[0] if len(x) > 0 else None), _as_scalar(x[1] if len(x) > 1 else None)
        if isinstance(x, _pd.Series):
            vals = list(x.values)
            return _as_scalar(vals[0] if len(vals) > 0 else None), _as_scalar(vals[1] if len(vals) > 1 else None)
        return None, None

    if not exact_nm:
        return (None, None)

    hit = id_map.get(str(exact_nm).strip())
    emp, stf = _to_pair(hit)
    if (emp is not None) or (stf is not None):
        return emp, stf

    nk = _name_key(exact_nm)
    for k, pair in id_map.items():
        if _name_key(k) == nk:
            return _to_pair(pair)
    return (None, None)


# ---------- core (heavy) ----------
def _get_people(selected_months: List[str], terminal: str | None) -> List[Dict[str, str]]:
    t0 = time.time()

    tnorm = (terminal or "ALL").upper()
    if tnorm not in {"ALL","P123","P456"}:
        tnorm = "ALL"

    month_key = tuple(sorted(selected_months)) if selected_months else ("__ALL__",)
    cache_key = (month_key, tnorm)
    now = _now()
    hit = _PEOPLE_CACHE.get(cache_key)
    if hit and (now - float(hit.get("ts", 0))) <= _PEOPLE_TTL:
        return hit.get("people", [])  # type: ignore[return-value]

    selected_months = _norm_keep_months(selected_months)

    tables = _main_tables()
    safety_tables = _safety_tables()

    frames: List[pd.DataFrame] = []
    for key_df in ("appended","p123_enriched","p456_enriched","p456_group_enriched","p123_ywt","p123_long"):
        df = tables.get(key_df)  # type: ignore[index]
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)

    if not frames:
        _PEOPLE_CACHE[cache_key] = {"ts": now, "people": []}
        return []

    def _build_base():
        b = pd.concat(frames, ignore_index=True)
        return _ensure_month_3(b)

    base = _get_cached_base(_build_base).copy()

    name_c  = _find_col(base, "Name")
    group_c = _find_col(base, "Group")
    plant_c = _find_col(base, "Plant","Source","Terminal","terminal","PLANT")
    date_c  = _find_col(base, "EVENT_SHIFT_DT", "DATE", "Date")
    shift_c = _find_col(base, "EVENT_HR12_SHIFT_C", "Shift")

    if not name_c:
        _PEOPLE_CACHE[cache_key] = {"ts": now, "people": []}
        return []

    base[name_c] = base[name_c].astype(str).str.strip()
    base = base[base[name_c].ne("")]
    if group_c:
        base = _filter_to_abcd(base, group_c)

    # remove Rahimi (no warning)
    base = base[~base[name_c].astype(str).str.contains(_RAHIMI_RX, case=False, regex=True, na=False)]

    if selected_months:
        keep = set(selected_months)
        base["Month"] = base["Month"].astype(str).str[:3]
        base = base[base["Month"].isin(keep)]

    base["_PL"] = base[plant_c].map(_which_plant).fillna("P123") if plant_c else "P123"
    if tnorm in {"P123","P456"}:
        base = base[base["_PL"] == tnorm]

    base["_norm"] = base[name_c].astype(str).map(_name_key)

    # plant per person (used for terminal-specific YWT targets, even when terminal=ALL)
    try:
        _mode = base.groupby("_norm")["_PL"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        plant_by_person = _mode.to_dict()
    except Exception:
        plant_by_person = {}

    if date_c and shift_c:
        base_keys = _shift_key(base, date_c, shift_c)
    else:
        base_keys = pd.Series([pd.NA] * len(base))

    worked_df = pd.DataFrame({"_norm": base["_norm"], "_key": base_keys}).dropna(subset=["_norm"]).drop_duplicates()
    worked_df = worked_df.dropna(subset=["_key"])
    by_person_shifts = worked_df.groupby("_norm")["_key"].nunique()
    allowed_keys = set(worked_df["_key"])

    print("[T] base build", round(time.time()-t0, 3)); t0=time.time()

    # ======= SAFETY =======
    safety_bundle = safety_tables
    safety_df = _get_table_ci(safety_bundle, "safety")
    if not isinstance(safety_df, pd.DataFrame) or safety_df.empty:
        frames_si = []
        for df_any in safety_bundle.values():
            if isinstance(df_any, pd.DataFrame) and any("si" in str(c).lower() for c in df_any.columns):
                frames_si.append(df_any)
        safety_df = pd.concat(frames_si, ignore_index=True) if frames_si else None

    si_done_on_base = pd.Series(dtype=int)

    if isinstance(safety_df, pd.DataFrame) and not safety_df.empty:
        saf = _ensure_month_3(safety_df.copy())
        if selected_months:
            keep = set(selected_months)
            saf["Month"] = saf["Month"].astype(str).str[:3]
            saf = saf[saf["Month"].isin(keep)]

        name_col = _find_col(saf, "Name")
        if not name_col:
            key_col = _find_col(saf, "Key")
            if key_col:
                saf["_NameFromKey"] = saf[key_col].astype(str).str.split("-", n=1, expand=True)[0].str.strip()
                name_col = "_NameFromKey"

        si_col   = _find_si_col(saf)
        date_col = _find_col(saf, "EVENT_SHIFT_DT", "DATE", "Date")
        shift_col= _find_col(saf, "EVENT_HR12_SHIFT_C", "Shift")

        if name_col:
            saf["_norm_safety"] = saf[name_col].astype(str).map(_name_key)
        else:
            saf["_norm_safety"] = pd.NA

        raw_safety_names = saf[name_col].dropna().astype(str).unique().tolist() if name_col else []
        raw_base_names   = base[name_c].dropna().astype(str).unique().tolist()

        name_map = _get_cached_name_map(cache_key, raw_safety_names, raw_base_names)

        for s_norm in saf["_norm_safety"].dropna().unique().tolist():
            if s_norm not in name_map and s_norm in by_person_shifts.index:
                name_map[s_norm] = s_norm

        if name_col and date_col and shift_col:
            saf["_key"] = _shift_key(saf, date_col, shift_col)
            if si_col:
                v = saf[si_col]
                is_num_pos = pd.to_numeric(v, errors="coerce").fillna(0) > 0
                txt = v.astype(str).str.strip().str.upper()
                is_text_pos = ~txt.isin({"","0","N","NO","FALSE"})
                has_si = is_num_pos | is_text_pos
            else:
                has_si = pd.Series([True] * len(saf), index=saf.index)

            saf_ok = saf[has_si].dropna(subset=["_norm_safety","_key"]).drop_duplicates()
            try:
                saf_ok = saf_ok[saf_ok["_key"].isin(allowed_keys)]
            except Exception:
                pass

            saf_ok["_norm_base"] = saf_ok["_norm_safety"].map(name_map)
            saf_ok = saf_ok.dropna(subset=["_norm_base"])
            si_done_on_base = saf_ok.groupby("_norm_base")["_key"].nunique()
        else:
            if name_col and si_col:
                saf["_si"] = pd.to_numeric(saf[si_col], errors="coerce").fillna(0).astype(int).clip(lower=0)
                saf["_norm_base"] = saf["_norm_safety"].map(name_map)
                saf = saf.dropna(subset=["_norm_base"])
                si_done_on_base = saf.groupby("_norm_base")["_si"].sum()

    safety_pts_map: Dict[str, float] = {}
    safety_ratio_map: Dict[str, Tuple[int, int]] = {}

    for base_norm, shift_cnt in by_person_shifts.items():
        si_cnt = int(si_done_on_base.get(base_norm, 0))
        shift_cnt_i = int(shift_cnt)

        pts = round((si_cnt / shift_cnt_i), 2) if shift_cnt_i > 0 else 0.0  # <-- NEW RULE
        safety_pts_map[base_norm] = pts
        safety_ratio_map[base_norm] = (si_cnt, shift_cnt_i)


    print("[T] safety", round(time.time()-t0, 3)); t0=time.time()

    # ======= Operations (YWT) =======
    ywt_col = _find_col(base, "YWT")
    hsl_col_dummy = _find_col(base, "HSL")
    ywt_map: Dict[str, float] = {}
    if ywt_col:
        ytmp = pd.to_numeric(base[ywt_col], errors="coerce")
        ywt_map = base.assign(_y=ytmp).groupby("_norm")["_y"].mean().to_dict()

    # ======= HSL =======
    hsl_rates_df = _get_cached_hsl()

    # per-person mean pass rate (0..100) over worked shifts
    hsl_pct_map: Dict[str, float] = {}
    if isinstance(hsl_rates_df, pd.DataFrame) and not hsl_rates_df.empty:
        tmp = hsl_rates_df[hsl_rates_df["_key"].isin(allowed_keys)][["_key", "pass_rate"]].copy()
        merged = worked_df.merge(tmp, on="_key", how="left")
        per = merged.groupby("_norm")["pass_rate"].mean()
        hsl_pct_map = per.to_dict()


    def ywt_points_for(v: float | None, plant: str | None) -> Tuple[float, str]:
        """Terminal-specific YWT scoring.
        Base points:
          P123 target 9.7: <9.7=5, 9.7-9.9=4, 9.9-10.1=3, 10.1-10.3=2, else=1
          P456 target 7.4: <7.4=5, 7.4-7.6=4, 7.6-7.8=3, 7.8-8.0=2, else=1
        Final YWT points = base_points * 0.4
        """
        if v is None or pd.isna(v):
            return 0.0, "pts-red"

        p = (plant or "").upper()
        if p == "P456":
            target = 7.4
        else:
            # default to P123 rules
            target = 9.7

        x = float(v)
        if x < target:
            base_pts = 5
        elif x < target + 0.2:
            base_pts = 4
        elif x < target + 0.4:
            base_pts = 3
        elif x < target + 0.6:
            base_pts = 2
        else:
            base_pts = 1

        pts = round(float(base_pts) * 0.4, 2)

        if base_pts >= 4:
            cls = "pts-green"
        elif base_pts == 3:
            cls = "pts-orange"
        else:
            cls = "pts-red"

        return pts, cls

        # dedup people (IMPORTANT: include "_norm" and DO NOT use row._norm attribute)
    role_c = _find_col(base, "Role")
    dedup_cols = [name_c, "_norm"]
    if group_c:
        dedup_cols.append(group_c)
    if role_c:
        dedup_cols.append(role_c)

    dedup = base[dedup_cols].drop_duplicates(subset=[name_c], keep="first")

    col_i = {c: i for i, c in enumerate(dedup.columns)}
    idx_name = col_i[name_c]
    idx_norm = col_i["_norm"]
    idx_group = col_i[group_c] if group_c and group_c in col_i else None
    idx_role = col_i[role_c] if role_c and role_c in col_i else None

    photos = _photos_dict()
    people: List[Dict[str, str]] = []

    for idx, row in enumerate(dedup.itertuples(index=False, name=None)):
        raw_name = str(row[idx_name]).strip()
        nm_key = str(row[idx_norm]).strip() if row[idx_norm] is not None else _name_key(raw_name)

        group_val = str(row[idx_group]).strip() if idx_group is not None else ""
        fallback_desig = str(row[idx_role]).strip() if idx_role is not None else ""

        manual_desig = _designation_for(raw_name)
        desig = manual_desig or fallback_desig

        # SAFETY
        s_pts = round(float(safety_pts_map.get(nm_key, 0.0)), 2)
        si_cnt, shift_cnt = safety_ratio_map.get(nm_key, (0, 0))
        s_cls  = _safety_class(s_pts)

        # POSITIVE CONTRIBUTION (still demo)
        pat = idx % 6
        if pat in (0, 1):
            pc_count = 4
        elif pat in (2, 3):
            pc_count = 3
        else:
            pc_count = 1

        # NEW points rule: count * 0.1, cap at 0.5
        pc_points = round(min(0.5, float(pc_count) * 0.1), 2)

        # optional class (tweak if you want)
        if pc_points >= 0.4:
            pc_class = "pts-green"
        elif pc_points >= 0.2:
            pc_class = "pts-orange"
        else:
            pc_class = "pts-red"

        # OPERATIONS YWT
        raw_ywt = ywt_map.get(nm_key)
        plant_for_person = plant_by_person.get(nm_key) or (tnorm if tnorm in {"P123","P456"} else "P123")
        ywt_pts, ywt_class = ywt_points_for(raw_ywt, plant_for_person)
        ywt_display = f"{raw_ywt:.2f}" if raw_ywt is not None and not pd.isna(raw_ywt) else "-"

        # HSL (points like reliability: pct * 0.005, cap 0.5)
        extracted_hsl = hsl_pct_map.get(nm_key, float("nan"))
        if not pd.isna(extracted_hsl):
            hsl_display = f"{float(extracted_hsl):.1f}%"
            hsl_pts = _pct_to_half_point(extracted_hsl)
        else:
            hsl_display = "-"
            hsl_pts = 0.0
        hsl_class = _half_point_class(hsl_pts, orange_at=0.43, green_at=0.45)

        total_points = float(s_pts) + float(pc_points) + float(ywt_pts) + float(hsl_pts)
        total_points = round(total_points, 2)

        people.append({
            "name": _display_name(raw_name),
            "group": group_val,
            "designation": desig,

            "safety_points": s_pts,
            "safety_class": s_cls,
            "si_done": si_cnt,
            "shift_count": shift_cnt,

            "pc_count": pc_count,
            "pc_points": pc_points,
            "pc_class": pc_class,

            "ywt": ywt_display,
            "ywt_points": ywt_pts,
            "ywt_class": ywt_class,

            "hsl": hsl_display,
            "hsl_points": hsl_pts,
            "hsl_class": hsl_class,

            "points": total_points,

            "photo": photos.get(nm_key, ""),  # use normalized key directly
            "_exact_name": raw_name,
        })

    print("[T] hsl", round(time.time()-t0, 3)); t0=time.time()

    # ====== RELIABILITY (attendance %) ======
    if callable(get_attendance_tables) and len(people) > 0:
        att_month = (selected_months[0][:3].title() if selected_months else "auto")
        id_map = _extract_ids_map(base)

        for p in people:
            exact_nm = str(p.get("_exact_name") or p.get("name") or "").strip() or None
            emp_id, staff_no = _id_lookup_with_fallback(id_map, exact_nm)
            attendance_stats = _attendance_cached(exact_nm, emp_id, staff_no, att_month)

            mc_used, ul_used = _resolve_mc_ul(attendance_stats, selected_months)
            shift_worked = int(p.get("shift_count") or 0)

            pct = None
            if mc_used is not None and ul_used is not None:
                denom = float(shift_worked) + float(mc_used) + float(ul_used)
                if denom > 0:
                    pct = max(0.0, min(100.0, 100.0 * (float(shift_worked) / denom)))

            p["reliability"] = (round(float(pct), 1) if isinstance(pct, (int, float)) and not pd.isna(pct) else 0.0)

        # NEW: reliability points = (reliability% / 100) * 5 * 0.1  (cap at 0.5)
        for p in people:
            r = float(p.get("reliability", 0.0) or 0.0)  # 0..100
            pts = round(min(0.5, (r / 100.0) * 5.0 * 0.1), 2)  # == min(0.5, r * 0.005)

            if pts >= 0.45:
                cls = "pts-green"
            elif pts >= 0.4:
                cls = "pts-orange"
            else:
                cls = "pts-red"

            p["reliability_points"] = pts
            p["reliability_class"] = cls

    else:
        for p in people:
            p["reliability"] = 0.0
            p["reliability_points"] = 0.0
            p["reliability_class"] = "pts-red"

    print("[T] attendance", round(time.time()-t0, 3)); t0=time.time()

    # ====== EXCELLENCE (demo) ======
    n = len(people)
    if n > 0:
        # demo rule: excellence is binary (1/0)
        # here: tie it to pc_count just to have a stable mix of 1s and 0s
        for person in people:
            pc_like = int(person.get("pc_count", 0) or 0)
            person["excellence"] = 1 if pc_like >= 3 else 0

            ex = int(person.get("excellence", 0) or 0)
            pts = round(float(ex) * 0.5, 2)  # 0.5 or 0.0
            person["excellence_points"] = pts
            person["excellence_class"] = "pts-green" if pts >= 0.5 else "pts-red"
    else:
        for person in people:
            person["excellence"] = 0
            person["excellence_points"] = 0.0
            person["excellence_class"] = "pts-red"

    # FINAL: recompute points from parts (prevents double-add / rounding drift)
    for person in people:
        person["points"] = round(
            float(person.get("safety_points", 0.0) or 0.0)
            + float(person.get("pc_points", 0.0) or 0.0)
            + float(person.get("ywt_points", 0.0) or 0.0)
            + float(person.get("hsl_points", 0.0) or 0.0)
            + float(person.get("reliability_points", 0.0) or 0.0)
            + float(person.get("excellence_points", 0.0) or 0.0),
            2,
        )

    people.sort(key=lambda x: (-float(x.get("points", 0.0)), x["name"]))
    _PEOPLE_CACHE[cache_key] = {"ts": now, "people": people}
    print("[T] Excellence", round(time.time()-t0, 3)); t0=time.time()
    return people

    return people

def _months_dropdown() -> List[str]:
    now = _now()
    if _MONTHS_CACHE["months"] and (now - float(_MONTHS_CACHE["ts"])) <= _MONTHS_TTL:
        return _MONTHS_CACHE["months"]  # type: ignore[return-value]

    months_seen: List[str] = []

    for df in _main_tables().values():
        if isinstance(df, pd.DataFrame) and not df.empty and "Month" in df.columns:
            months_seen.extend(df["Month"].astype(str).str[:3].unique().tolist())

    for df in _safety_tables().values():
        if isinstance(df, pd.DataFrame) and not df.empty and "Month" in df.columns:
            months_seen.extend(df["Month"].astype(str).str[:3].unique().tolist())

    months = _order_months(pd.unique(pd.Series(months_seen))) or MONTHS_ORDER
    _MONTHS_CACHE.update({"ts": now, "months": months})
    return months


# ---------- route ----------
@main.route("/main")
def index():
    raw_months = request.args.getlist("months") or request.args.getlist("month")
    selected_months = _norm_keep_months(raw_months)

    terminal = (request.args.get("terminal") or "ALL").upper()
    if terminal not in {"ALL","P123","P456"}:
        terminal = "ALL"

    months = _months_dropdown()
    people = _get_people(selected_months, terminal)

    return render_template(
        "main.html",
        months=months,
        selected_months=selected_months,
        people=people,
        selected_terminal=terminal,
    )
