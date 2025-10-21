# routes/main.py
from __future__ import annotations

import os, re, time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Blueprint, render_template, request

# ===== Primary data (P123/P456/appended/etc.) =====
from build.build_data import get_tables as get_main_tables  # type: ignore

# ===== Safety bundle (your project exposes Safety here) =====
from build.build_safety import get_tables as get_safety_tables  # type: ignore

# ===== Optional staff photos (if present) =====
try:
    from build.photos import get_tables as get_photo_tables  # type: ignore
except Exception:
    get_photo_tables = None

main = Blueprint("main", __name__, template_folder="templates")

# ---------- caches ----------
_CACHE_TTL = 120
_MAIN_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}
_SAFETY_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}
_PHOTO_TTL = 300
_PHOTO_CACHE: Dict[str, object] = {"ts": 0.0, "photos": {}}

PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"

MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- tiny utils ----------
def _now() -> float: return time.time()

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
    if df is None or df.empty: return None
    index = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_colname(cand)
        if k in index: return index[k]
    for alt in ("group","groupcode","empgroup","grp","terminal","plant","source"):
        if alt in index: return index[alt]
    return None

def _ensure_month_3(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if "Month" not in df.columns:
        dc = _find_col(df, "EVENT_SHIFT_DT","DATE","Date")
        if dc:
            dt = pd.to_datetime(df[dc], errors="coerce")
            df = df.copy(); df["Month"] = dt.dt.strftime("%b")
        else:
            df = df.copy(); df["Month"] = pd.NA
    else:
        df = df.copy(); df["Month"] = df["Month"].astype(str).str[:3]
    return df

def _filter_to_abcd(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    col = df[group_col].astype(str).str.strip().str.upper()
    uses_star = col.str.match(r"^[ABCD]\*$").any()
    mask = col.isin({"A*","B*","C*","D*"}) if uses_star else col.str.match(r"^[ABCD]")
    out = df[mask]
    if out.empty and not col.empty:
        out = df[col.str.contains(r"^\s*[ABCD]\s*\*?\s*$", regex=True)]
    return out

def _guess_wait_col(df: pd.DataFrame) -> str | None:
    cands = [c for c in df.columns if any(k in c.lower() for k in ("ywt","wait","avg"))]
    for c in cands:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return cands[0] if cands else None

def _which_plant(val: str) -> str | None:
    v = (val or "").upper()
    if "P456" in v or v.strip() == "456": return "P456"
    if "P123" in v or v.strip() == "123": return "P123"
    if v in {"P123","P456"}: return v
    return None

def _main_tables() -> Dict[str, pd.DataFrame]:
    now = _now()
    if (not _MAIN_CACHE["tables"]) or (now - float(_MAIN_CACHE["ts"])) > _CACHE_TTL:
        _MAIN_CACHE["tables"] = get_main_tables(show_errors=True)
        _MAIN_CACHE["ts"] = now
    return _MAIN_CACHE["tables"] or {}

def _safety_tables() -> Dict[str, pd.DataFrame]:
    """
    Return whatever tables build_safety exposes (should include 'Safety').
    Also merges main tables so we can still scan for SI if needed.
    """
    out: Dict[str, pd.DataFrame] = {}
    # 1) primary: build_safety
    try:
        bundle = get_safety_tables()
        if isinstance(bundle, dict):
            out.update(bundle)
    except Exception as e:
        print("[SFT] build_safety.get_tables() error:", e)

    # 2) also include main tables (as fallback scanning surface)
    try:
        for k, v in _main_tables().items():
            if isinstance(v, pd.DataFrame) and k not in out:
                out[k] = v
    except Exception:
        pass

    print("[SFT] safety bundle keys:", list(out.keys())[:12])
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

# ---------- YWT points ----------
def _ywt_points_for_plant(avg_wait: float, plant: str) -> int:
    if pd.isna(avg_wait): return 0
    x = float(avg_wait)
    if plant == "P123":
        if x < 9.5: return 6
        if 9.5 <= x < 9.7: return 4
        if 9.7 <= x < 10:  return 2
        return 0
    if plant == "P456":
        if x < 7.4: return 6
        if 7.4 <= x < 7.6: return 4
        if 7.6 <= x < 7.9: return 2
        return 0
    return 0

def _ywt_class(pts: int) -> str:
    if pts == 6: return "pts-green"
    if pts == 4: return "pts-orange"
    return "pts-red"

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

# ---------- core ----------
def _get_people(selected_months: List[str]) -> List[Dict[str, str]]:
    tables = _main_tables()
    safety_tables = _safety_tables()

    # choose base frames
    frames: List[pd.DataFrame] = []
    for key in ("appended","p123_enriched","p456_enriched","p456_group_enriched","p123_ywt","p123_long"):
        df = tables.get(key)  # type: ignore[index]
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)
    if not frames: return []

    base = pd.concat(frames, ignore_index=True)
    base = _ensure_month_3(base)

    # columns
    name_c  = _find_col(base, "Name")
    group_c = _find_col(base, "Group")
    plant_c = _find_col(base, "Plant","Source","Terminal","terminal","PLANT")
    wait_c  = _guess_wait_col(base)
    date_c  = _find_col(base, "EVENT_SHIFT_DT", "DATE", "Date")
    shift_c = _find_col(base, "EVENT_HR12_SHIFT_C", "Shift")
    if not name_c: return []

    base[name_c] = base[name_c].astype(str).str.strip()
    base = base[base[name_c].ne("")]
    if group_c: base = _filter_to_abcd(base, group_c)
    if selected_months:
        keep = [m[:3] for m in selected_months]
        base = base[base["Month"].astype(str).str[:3].isin(keep)]

    # YWT means/points
    if wait_c:
        base[wait_c] = pd.to_numeric(base[wait_c], errors="coerce")
    if not plant_c:
        base = base.copy(); base["_PLANT"] = "P123"; plant_c = "_PLANT"
    else:
        base = base.copy(); base[plant_c] = base[plant_c].astype(str)

    ywt_overall = base.groupby(name_c, dropna=False)[wait_c].mean() if wait_c else pd.Series(dtype=float)
    pts_map: Dict[str, int] = {}
    if wait_c:
        sub = base.dropna(subset=[wait_c]).copy()
        sub["_PL"] = sub[plant_c].map(_which_plant).fillna("P123")
        per_pp = sub.groupby([name_c,"_PL"], dropna=False)[wait_c].mean().reset_index()
        for _, r in per_pp.iterrows():
            nm = str(r[name_c]).strip()
            pl = str(r["_PL"])
            pts_map[nm] = pts_map.get(nm, 0) + _ywt_points_for_plant(r[wait_c], pl)

    # ===== SAFETY (robust to aggregated sheet + fuzzy names) =====
    DEBUG_SAFETY = False  # set True to see prints in console

    def _norm_shift(val: str) -> str:
        v = (val or "").strip().upper()
        if v.startswith("D"): return "D"
        if v.startswith("N"): return "N"
        return v[:1] if v else ""

    def _shift_key(df: pd.DataFrame, dc: str | None, sc: str | None) -> pd.Series:
        if not dc or not sc: return pd.Series([pd.NA] * len(df))
        dt = pd.to_datetime(df[dc], errors="coerce")
        sh = df[sc].astype(str).map(_norm_shift)
        return dt.dt.strftime("%Y-%m-%d") + "@" + sh

    def _find_si_col(df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if str(c).strip().lower() == "si": return c
        for c in df.columns:
            if "si" in str(c).strip().lower(): return c
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.5 and (s.fillna(0) == s.fillna(0).round()).mean() > 0.9:
                return c
        return None

    def _token_list(name: str) -> list[str]: return [t for t in _name_key(name).split() if t]
    def _token_set(name: str) -> set[str]:   return set(_token_list(name))

    def _name_similarity(a: str, b: str) -> float:
        A, B = _token_set(a), _token_set(b)
        if not A or not B: return 0.0
        inter = len(A & B)
        jac = inter / len(A | B)
        a_n, b_n = _name_key(a), _name_key(b)
        if a_n in b_n or b_n in a_n: jac += 0.4
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
            if s_norm in mapping: continue
            toks = _token_list(s)[:2]
            if not toks: continue
            for b in base_raw:
                if set(toks).issubset(_token_set(b)):
                    mapping[s_norm] = _name_key(b)
                    break
        for s in safety_raw:
            s_norm = _name_key(s)
            if s_norm in mapping: continue
            for b in base_raw:
                bn = _name_key(b)
                if s_norm in bn or bn in s_norm:
                    mapping[s_norm] = bn
                    break
        return mapping

    # worked shifts (unique date+shift) from base
    base_keys = _shift_key(base, date_c, shift_c)
    worked_df = pd.DataFrame({"_norm": base[name_c].astype(str).map(_name_key),
                              "_key": base_keys})
    worked_df = worked_df.dropna(subset=["_norm","_key"]).drop_duplicates()
    by_person_shifts = worked_df.groupby("_norm")["_key"].nunique()  # index: base_norm_name

    # safety dataframe (from build_safety)
    safety_bundle = _safety_tables()
    safety_df = _get_table_ci(safety_bundle, "safety")
    if not isinstance(safety_df, pd.DataFrame) or safety_df.empty:
        # scan any table for an SI column as last resort
        frames_si = []
        for df_any in safety_bundle.values():
            if isinstance(df_any, pd.DataFrame) and any("si" in str(c).lower() for c in df_any.columns):
                frames_si.append(df_any)
        safety_df = pd.concat(frames_si, ignore_index=True) if frames_si else None

    si_done_on_base = pd.Series(dtype=int)  # index: base_norm_name

    if isinstance(safety_df, pd.DataFrame) and not safety_df.empty:
        saf = _ensure_month_3(safety_df.copy())
        if DEBUG_SAFETY:
            print("[SFT] raw safety shape:", saf.shape)
            print("[SFT] raw safety cols:", list(saf.columns))

        if selected_months:
            keep = [m[:3] for m in selected_months]
            saf = saf[saf["Month"].astype(str).str[:3].isin(keep)]

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

        # per-shift rows
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
            saf_ok["_norm_base"] = saf_ok["_norm_safety"].map(name_map)
            saf_ok = saf_ok.dropna(subset=["_norm_base"])
            si_done_on_base = saf_ok.groupby("_norm_base")["_key"].nunique()
        # aggregated per person/month (your sheet)
        else:
            if name_col and si_col:
                saf["_si"] = pd.to_numeric(saf[si_col], errors="coerce").fillna(0).astype(int).clip(lower=0)
                saf["_norm_base"] = saf["_norm_safety"].map(name_map)
                saf = saf.dropna(subset=["_norm_base"])
                si_done_on_base = saf.groupby("_norm_base")["_si"].sum()

        if DEBUG_SAFETY:
            print("[SFT] worked_shifts (first 8):", list(by_person_shifts.items())[:8])
            print("[SFT] si_done_on_base (first 8):", list(si_done_on_base.items())[:8])

    # safety points + ratios
    safety_pts_map: Dict[str, int] = {}
    safety_ratio_map: Dict[str, Tuple[int,int]] = {}
    for base_norm, shift_cnt in by_person_shifts.items():
        si_cnt = int(si_done_on_base.get(base_norm, 0))
        pts = 3 if (shift_cnt > 0 and si_cnt == shift_cnt) else 0
        safety_pts_map[base_norm] = pts
        safety_ratio_map[base_norm] = (si_cnt, int(shift_cnt))

    # dedup people
    role_c = _find_col(base, "Role")
    cols = [name_c] + ([group_c] if group_c else []) + ([role_c] if role_c else [])
    dedup = base[cols].drop_duplicates(subset=[name_c], keep="first")

    photos = _photos_dict()
    people: List[Dict[str, str]] = []
    for _, r in dedup.iterrows():
        raw_name = str(r[name_c]).strip()
        nm_key = _name_key(raw_name)
        group_val = str(r.get(group_c, "") or "")

        manual_desig = _designation_for(raw_name)
        fallback_desig = str(r.get(role_c, "") or "")
        desig = manual_desig or fallback_desig

        mean_wait = float(ywt_overall.get(raw_name)) if wait_c and raw_name in ywt_overall else float("nan")
        ywt_disp = "-" if pd.isna(mean_wait) else f"{mean_wait:.2f}"
        ywt_pts  = int(pts_map.get(raw_name, 0))
        ywt_cls  = _ywt_class(ywt_pts)

        s_pts  = int(safety_pts_map.get(nm_key, 0))
        si_cnt, shift_cnt = safety_ratio_map.get(nm_key, (0, 0))
        s_cls  = _safety_class(s_pts)

        people.append({
            "name": _display_name(raw_name),
            "group": group_val,
            "designation": desig,

            "ywt": ywt_disp,
            "ywt_points": ywt_pts,
            "ywt_class": ywt_cls,

            "safety_points": s_pts,
            "safety_class": s_cls,
            "si_done": si_cnt,
            "shift_count": shift_cnt,

            "hsl": "-",
            "fish": "-",

            "points": (ywt_pts + s_pts),

            "photo": photos.get(_name_key(raw_name), ""),
        })

    people.sort(key=lambda x: (-int(x["points"]), x["name"]))
    return people

@main.route("/main")
def index():
    selected_months = request.args.getlist("month")

    months_seen: List[str] = []
    for df in _main_tables().values():
        if isinstance(df, pd.DataFrame) and not df.empty and "Month" in df.columns:
            months_seen.extend(df["Month"].astype(str).str[:3].unique().tolist())
    months = _order_months(pd.unique(pd.Series(months_seen))) or MONTHS_ORDER

    people = _get_people(selected_months)

    return render_template(
        "main.html",
        months=months,
        selected_months=selected_months,
        people=people,
    )
