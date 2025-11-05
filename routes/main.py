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

# NEW: import HSL tables (same style as attached file)
from build.build_H import get_tables as get_hsl_tables  # type: ignore

main = Blueprint("main", __name__, template_folder="templates")

# ---------- caches ----------
_CACHE_TTL = 600
_MAIN_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}
_SAFETY_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}
_PHOTO_TTL = 300
_PHOTO_CACHE: Dict[str, object] = {"ts": 0.0, "photos": {}}
_PEOPLE_TTL = 300
_PEOPLE_CACHE: Dict[Tuple[Tuple[str, ...], str], Dict[str, object]] = {}
_MONTHS_TTL = 600
_MONTHS_CACHE: Dict[str, object] = {"ts": 0.0, "months": None}

PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"
MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ---------- tiny utils ----------
def _now() -> float:
    return time.time()


def _order_months(ms) -> List[str]:
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s]


def _name_key(s) -> str:
    t = "" if pd.isna(s) else str(s)
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
    # common fallbacks
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


def _which_plant(val: str) -> str | None:
    v = (val or "").upper()
    if "P456" in v or v.strip() == "456":
        return "P456"
    if "P123" in v or v.strip() == "123":
        return "P123"
    if v in {"P123","P456"}:
        return v
    return None


# --- month normalization ---
def _to_month_abbr(x) -> str:
    """Cheap per-value month normalizer (no pandas construction)."""
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


def _pick_df(bundle: dict, *keys: str) -> pd.DataFrame | None:
    if not isinstance(bundle, dict):
        return None
    for k in keys:
        v = bundle.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
    for k in keys:
        v = bundle.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


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
        date_only = g[date_col].astype(str).str.replace(r"\s+[DNdn]$", "", regex=True)
        g["_sd_date"] = _fast_parse_date(date_only, dayfirst=True).dt.date
        missing = g["_sd_shift"].isna() | (g["_sd_shift"]=="")
        if missing.any():
            sh = g.loc[missing, date_col].astype(str).str.extract(r"([DNdn])$")[0].fillna("")
            g.loc[missing, "_sd_shift"] = sh.str.upper().map(_norm_shift_letter)
    else:
        g["_sd_date"] = _fast_parse_date(g[date_col], dayfirst=True).dt.date
    return g


def _pct_to_hsl_points(pct: float) -> int:
    if pd.isna(pct):
        return 0
    x = float(pct)
    if x > 90:
        return 3
    if x >= 85:
        return 2
    if x >= 80:
        return 1
    return 0


def _extract_hsl_rates_from_log() -> pd.DataFrame:
    """Read HSL log from build.build_H and turn into key=YYYY-MM-DD@Shift rows."""
    try:
        bundle = get_hsl_tables() or {}
    except Exception as e:
        print("[HSL] get_hsl_tables failed:", e)
        return pd.DataFrame(columns=["_key","pass_rate"])

    log_df = None
    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame) and str(k).strip().lower() == "log":
            log_df = v
            break
    if log_df is None or log_df.empty:
        print("[HSL] 'Log' not found or empty")
        return pd.DataFrame(columns=["_key","pass_rate"])

    g = _norm_df_shift_date_tuple(log_df).copy()
    g = g.dropna(subset=["_sd_date","_sd_shift"])
    g["_key"] = pd.to_datetime(g["_sd_date"]).dt.strftime("%Y-%m-%d") + "@" + g["_sd_shift"].astype(str)

    # try to find pass / total style cols
    col_good = None
    col_total = None
    for c in g.columns:
        lc = str(c).lower()
        if "good" in lc or "pass" in lc:
            col_good = c
        if "total" in lc:
            col_total = c

    rows = []
    if col_good and col_total:
        g["_good"] = pd.to_numeric(g[col_good], errors="coerce").fillna(0)
        g["_total"] = pd.to_numeric(g[col_total], errors="coerce").fillna(0)
        agg = g.groupby("_key")[["_good","_total"]].sum()
        for key, row in agg.iterrows():
            tot = float(row["_total"])
            if tot <= 0:
                continue
            pct = (float(row["_good"]) / tot) * 100.0
            rows.append({"_key": key, "pass_rate": pct})

    return pd.DataFrame(rows)


# ---------- caches fetchers ----------
def _main_tables() -> Dict[str, pd.DataFrame]:
    now = _now()
    if (not _MAIN_CACHE["tables"]) or (now - float(_MAIN_CACHE["ts"])) > _CACHE_TTL:
        _MAIN_CACHE["tables"] = get_main_tables(show_errors=True)
        _MAIN_CACHE["ts"] = now
    return _MAIN_CACHE["tables"] or {}


def _safety_tables() -> Dict[str, pd.DataFrame]:
    # merge safety bundle with main bundle fallbacks
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
    return out


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
    return "pts-green" if pts == 3 else "pts-red"


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


def _norm_shift(val: str) -> str:
    v = (val or "").strip().upper()
    if v.startswith("D"):
        return "D"
    if v.startswith("N"):
        return "N"
    return v[:1] if v else ""


def _shift_key(df: pd.DataFrame, dc: str | None, sc: str | None) -> pd.Series:
    if not dc or not sc:
        return pd.Series([pd.NA] * len(df))
    dt = _fast_parse_date(df[dc], dayfirst=False)
    sh = df[sc].astype(str).map(_norm_shift)
    return dt.dt.strftime("%Y-%m-%d") + "@" + sh


def _find_si_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "si":
            return c
    for c in df.columns:
        if "si" in str(c).strip().lower():
            return c
    # last fallback: try a numeric 0/1 looking col
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
    # primary pass
    for s in safety_raw:
        s_norm = _name_key(s)
        best_norm, best_sim = None, -1.0
        for b in base_raw:
            sim = _name_similarity(s, b)
            if sim > best_sim:
                best_norm, best_sim = _name_key(b), sim
        if best_sim >= threshold and best_norm:
            mapping[s_norm] = best_norm
    # second pass
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
    # third pass
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


# ---------- core (heavy) ----------
def _get_people(selected_months: List[str], terminal: str | None) -> List[Dict[str, str]]:
    """
    Safety + dummy positive contribution + ops (if present) + reliability + excellence.
    """
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

    # base = your big combined main table
    frames: List[pd.DataFrame] = []
    for key_df in ("appended","p123_enriched","p456_enriched","p456_group_enriched","p123_ywt","p123_long"):
        df = tables.get(key_df)  # type: ignore[index]
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)

    if not frames:
        _PEOPLE_CACHE[cache_key] = {"ts": now, "people": []}
        return []

    base = pd.concat(frames, ignore_index=True)
    base = _ensure_month_3(base)

    name_c  = _find_col(base, "Name")
    group_c = _find_col(base, "Group")
    plant_c = _find_col(base, "Plant","Source","Terminal","terminal","PLANT")
    date_c  = _find_col(base, "EVENT_SHIFT_DT", "DATE", "Date")
    shift_c = _find_col(base, "EVENT_HR12_SHIFT_C", "Shift")

    if not name_c:
        _PEOPLE_CACHE[cache_key] = {"ts": now, "people": []}
        return []

    # clean & filter
    base[name_c] = base[name_c].astype(str).str.strip()
    base = base[base[name_c].ne("")]
    if group_c:
        base = _filter_to_abcd(base, group_c)

    # remove Rahimi
    base = base[~base[name_c].astype(str).str.contains(r"(^|\s)rahimi(\s|$)", case=False, regex=True)]

    # month filter
    if selected_months:
        keep = set(selected_months)
        base["Month"] = base["Month"].astype(str).str[:3]
        base = base[base["Month"].isin(keep)]

    # terminal filter
    base["_PL"] = base[plant_c].map(_which_plant).fillna("P123") if plant_c else "P123"
    if tnorm in {"P123","P456"}:
        base = base[base["_PL"] == tnorm]

    # make a norm col early
    base["_norm"] = base[name_c].astype(str).map(_name_key)

    # ---- worked shifts per person (denominator) ----
    if date_c and shift_c:
        base_keys = _shift_key(base, date_c, shift_c)
    else:
        base_keys = pd.Series([pd.NA] * len(base))

    worked_df = pd.DataFrame({
        "_norm": base["_norm"],
        "_key":  base_keys
    }).dropna(subset=["_norm"]).drop_duplicates()

    worked_df = worked_df.dropna(subset=["_key"])
    by_person_shifts = worked_df.groupby("_norm")["_key"].nunique()
    allowed_keys = set(worked_df["_key"])

    # ======= SAFETY load (same working version) =======
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
        name_map = _build_name_map(raw_safety_names, raw_base_names, threshold=0.5)

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

    # build safety maps
    safety_pts_map: Dict[str, int] = {}
    safety_ratio_map: Dict[str, Tuple[int,int]] = {}
    for base_norm, shift_cnt in by_person_shifts.items():
        si_cnt = int(si_done_on_base.get(base_norm, 0))
        pts = 3 if (shift_cnt > 0 and si_cnt == shift_cnt) else 0
        safety_pts_map[base_norm] = pts
        safety_ratio_map[base_norm] = (si_cnt, int(shift_cnt))

    # ======= Operations (pull if present) =======
    ywt_col = _find_col(base, "YWT")
    hsl_col_dummy = _find_col(base, "HSL")  # in case some ops table already has HSL %
    ywt_map: Dict[str, float] = {}
    if ywt_col:
        ytmp = pd.to_numeric(base[ywt_col], errors="coerce")
        ywt_map = base.assign(_y=ytmp).groupby("_norm")["_y"].mean().to_dict()

    # ======= HSL from separate build =======
    hsl_rates_df = _extract_hsl_rates_from_log()
    hsl_avg_by_key = pd.Series(dtype=float)
    if not hsl_rates_df.empty:
        hsl_rates_df = hsl_rates_df[hsl_rates_df["_key"].isin(allowed_keys)]
        hsl_avg_by_key = hsl_rates_df.groupby("_key")["pass_rate"].mean()

    hsl_pct_map: Dict[str, float] = {}
    hsl_pts_map: Dict[str, int] = {}
    if not hsl_avg_by_key.empty:
        keys_by_person = worked_df.groupby("_norm")["_key"].apply(list)
        for base_norm, keys in keys_by_person.items():
            pcts = [float(hsl_avg_by_key.get(k)) for k in keys if k in hsl_avg_by_key.index]
            pcts = [x for x in pcts if pd.notna(x)]
            avg_pct = (sum(pcts)/len(pcts)) if pcts else float("nan")
            hsl_pct_map[base_norm] = avg_pct
            hsl_pts_map[base_norm] = _pct_to_hsl_points(avg_pct)

    def ywt_points_for(v: float | None) -> Tuple[int, str]:
        if v is None or pd.isna(v):
            return 0, "pts-red"
        if v <= 7.30:
            return 6, "pts-green"
        elif v <= 7.40:
            return 4, "pts-green"
        elif v <= 7.50:
            return 3, "pts-orange"
        else:
            return 2, "pts-red"

    def hsl_points_for_existing(v: float | None) -> Tuple[int, str]:
        if v is None or pd.isna(v):
            return 0, "pts-red"
        if v >= 90:
            return 3, "pts-green"
        elif v >= 85:
            return 2, "pts-orange"
        else:
            return 0, "pts-red"

    # dedup people
    role_c = _find_col(base, "Role")
    cols = [name_c] + ([group_c] if group_c else []) + ([role_c] if role_c else [])
    dedup = base[cols].drop_duplicates(subset=[name_c], keep="first")

    photos = _photos_dict()
    people: List[Dict[str, str]] = []
    for idx, (_, r) in enumerate(dedup.iterrows()):
        raw_name = str(r[name_c]).strip()
        nm_key = _name_key(raw_name)
        group_val = str(r.get(group_c, "") or "")

        manual_desig = _designation_for(raw_name)
        fallback_desig = str(r.get(role_c, "") or "")
        desig = manual_desig or fallback_desig

        # SAFETY
        s_pts  = int(safety_pts_map.get(nm_key, 0))
        si_cnt, shift_cnt = safety_ratio_map.get(nm_key, (0, 0))
        s_cls  = _safety_class(s_pts)

        # POSITIVE CONTRIBUTION (dummy, patterned)
        pat = idx % 6
        if pat in (0, 1):
            pc_count = 4
            pc_points = 2
            pc_class = "pts-green"
        elif pat in (2, 3):
            pc_count = 3
            pc_points = 1
            pc_class = "pts-orange"
        else:
            pc_count = 1
            pc_points = 0
            pc_class = "pts-red"

        # OPERATIONS YWT
        raw_ywt = ywt_map.get(nm_key)
        ywt_pts, ywt_class = ywt_points_for(raw_ywt)
        ywt_display = f"{raw_ywt:.2f}" if raw_ywt is not None and not pd.isna(raw_ywt) else "-"

        # HSL
        extracted_hsl = hsl_pct_map.get(nm_key, float("nan"))
        if not pd.isna(extracted_hsl):
            hsl_display = f"{extracted_hsl:.1f}%"
            hsl_pts = hsl_pts_map.get(nm_key, 0)
        else:
            raw_hsl = None
            if hsl_col_dummy and hsl_col_dummy in base.columns:
                raw_hsl = None
            hsl_pts, _ = hsl_points_for_existing(raw_hsl if raw_hsl is not None else None)
            hsl_display = "-"

        total_points = s_pts + pc_points + ywt_pts + hsl_pts

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

            "points": total_points,
            "photo": photos.get(_name_key(raw_name), ""),
        })

    # ====== RELIABILITY (dummy, percentile → points) ======
    n = len(people)
    if n > 0:
        # make a wide-spread dummy reliability % for each person
        for i, person in enumerate(people):
            # 55..~100 with some variation
            base_val = 55 + (i * 17) % 45   # 55..99
            reliability = round(float(base_val) + ((i % 3) * 0.2), 1)
            person["reliability"] = reliability

        # rank by reliability descending
        order = sorted(range(n), key=lambda k: people[k]["reliability"], reverse=True)

        # percentile bands
        rel_bands = [
            (0.10, 5),
            (0.20, 4),
            (0.30, 3),
            (0.40, 2),
            (0.50, 1),
        ]

        for rank, idx_r in enumerate(order):
            pos = (rank + 1) / n
            pts = 0
            for cutoff, val in rel_bands:
                if pos <= cutoff:
                    pts = val
                    break

            if pts >= 5:
                cls = "pts-green"
            elif pts >= 3:
                cls = "pts-orange"
            elif pts >= 1:
                cls = "pts-red"
            else:
                cls = "pts-red"

            people[idx_r]["reliability_points"] = pts
            people[idx_r]["reliability_class"] = cls
            people[idx_r]["points"] = int(people[idx_r].get("points", 0)) + pts

        # ====== EXCELLENCE (dummy, percentile → points) ======
        # make dummy excellence similar to contribution but more spread out at the top
        for i, person in enumerate(people):
            pc_like = int(person.get("pc_count", 0))
            wiggle = (i * 13) % 9  # 0..8
            excellence_val = pc_like + wiggle
            person["excellence"] = excellence_val

        order_exc = sorted(range(n), key=lambda k: people[k]["excellence"], reverse=True)

        exc_bands = [
            (0.10, 5),
            (0.20, 4),
            (0.30, 3),
            (0.40, 2),
            (0.50, 1),
        ]

        for rank, idx_e in enumerate(order_exc):
            pos = (rank + 1) / n
            pts = 0
            for cutoff, val in exc_bands:
                if pos <= cutoff:
                    pts = val
                    break

            if pts >= 5:
                cls = "pts-green"
            elif pts >= 3:
                cls = "pts-orange"
            elif pts >= 1:
                cls = "pts-red"
            else:
                cls = "pts-red"

            people[idx_e]["excellence_points"] = pts
            people[idx_e]["excellence_class"] = cls
            people[idx_e]["points"] = int(people[idx_e].get("points", 0)) + pts

    # final sort
    people.sort(key=lambda x: (-int(x["points"]), x["name"]))

    _PEOPLE_CACHE[cache_key] = {"ts": now, "people": people}
    return people


def _months_dropdown() -> List[str]:
    """Compute months for the dropdown, cached."""
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
