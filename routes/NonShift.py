from flask import Blueprint, render_template, request, redirect, url_for
import time, os, re, difflib
import pandas as pd
from pathlib import Path

# DATA
from build.build_data import get_tables

# PHOTOS
from build.photos import get_tables as get_photo_tables

# SAFETY
try:
    from build.build_safety import get_tables as get_safety_tables
except Exception:
    get_safety_tables = None

nonshift = Blueprint("nonshift", __name__, url_prefix="/non-shift", template_folder="templates")

# ========= CONFIG =========
PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"

# ========= caches =========
_CACHE_TTL_SEC = 120
_CACHE = {"ts": 0.0, "tables": None}
_PHOTO_TTL_SEC = 300
_PHOTO_CACHE = {"ts": 0.0, "photos_dict": {}}
_SAFETY_TTL_SEC = 300
_SAFETY_CACHE = {"ts": 0.0, "df": None}

def _tables_cached():
    now = time.time()
    if (not _CACHE["tables"]) or (now - _CACHE["ts"] > _CACHE_TTL_SEC):
        _CACHE["tables"] = get_tables(show_errors=True)
        _CACHE["ts"] = now
    return _CACHE["tables"]

def _name_key(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _display_name(n: str) -> str:
    if not n: return n
    return re.sub(r"\bD O\b", "D/O", n, flags=re.IGNORECASE)

def _photos_cached():
    now = time.time()
    if (now - _PHOTO_CACHE["ts"] > _PHOTO_TTL_SEC) or not _PHOTO_CACHE["photos_dict"]:
        photos_dict = {}
        if os.path.isdir(PHOTO_DIR):
            try:
                bundle = get_photo_tables(PHOTO_DIR)  # {"photos_df": df, "photos_dict": dict}
                raw = (bundle or {}).get("photos_dict", {}) or {}
                photos_dict = { _name_key(k): v for k, v in raw.items() }
            except Exception:
                photos_dict = {}
        _PHOTO_CACHE.update({"ts": now, "photos_dict": photos_dict})
    return _PHOTO_CACHE["photos_dict"]

def _safety_df_cached():
    now = time.time()
    if (now - _SAFETY_CACHE["ts"] > _SAFETY_TTL_SEC) or _SAFETY_CACHE["df"] is None:
        df = None
        if callable(get_safety_tables):
            try:
                bundle = get_safety_tables()
                if isinstance(bundle, dict):
                    for k in ("safety", "Safety", "safety_df", "df", "table"):
                        v = bundle.get(k)
                        if isinstance(v, pd.DataFrame) and not v.empty:
                            df = v; break
            except Exception:
                df = None
        _SAFETY_CACHE.update({"ts": now, "df": df})
    return _SAFETY_CACHE["df"]

# ========= utils =========
MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _order_months(ms):
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s]

def _first_df(tbls: dict, *keys: str):
    for k in keys:
        df = tbls.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None

def _col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns: return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def _ensure_month_3(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" not in df.columns:
        date_col = _col(df, "EVENT_SHIFT_DT", "DATE", "Date", "Shift Date", "SDATE")
        if date_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            df = df.copy(); df["Month"] = dt.dt.strftime("%b")
        else:
            df = df.copy(); df["Month"] = pd.NA
    else:
        df = df.copy(); df["Month"] = df["Month"].astype(str).str[:3]
    return df
def _find_col(df: pd.DataFrame, *cands: str):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _guess_wait_col(df: pd.DataFrame):
    # prefer numeric columns that look like wait/ywt/avg
    pick = None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("ywt", "wait", "avg")):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
            pick = pick or c
    return pick

def _slice_tab(df: pd.DataFrame, tab: str) -> pd.DataFrame:
    plant_col = _col(df, "Plant", "PLANT", "Source")
    if plant_col:
        mask = df[plant_col].astype(str).str.upper().eq(tab.upper()) | df[plant_col].isna()
        return df.loc[mask].copy()
    return df

def _upper(s) -> str:
    return ("" if pd.isna(s) else str(s)).strip().upper()

def _make_cards(df: pd.DataFrame, photos_lookup: dict) -> list[dict]:
    """
    Build cards by NAME, but:
      - Only count shifts where the chosen YWT column is NOT NA
      - Average wait is computed only from those valid shifts
      - Other metrics (SI/NISR/SBR/Attendance/Stars) behave as before
    """
    if df.empty:
        return []

    df = _ensure_month_3(df)
    name_col  = _col(df, "Name", "NAME")
    group_col = _col(df, "group", "Group", "GROUP")
    si_col    = _col(df, "SI")
    nisr_col  = _col(df, "NISR")
    sbr_col   = _col(df, "SBR")
    att_col   = _col(df, "Attendance", "Attendance %")
    star_col  = _col(df, "Star", "STAR", "STAR_RATINGS", "Stars")
    wait_col  = _guess_wait_col(df)
    if not name_col:
        return []

    sdf = df.copy()
    sdf[name_col] = sdf[name_col].astype(str).str.strip()
    sdf = sdf[sdf[name_col].ne("")]

    # coerce numerics
    for c in [si_col, nisr_col, sbr_col, att_col, star_col, wait_col]:
        if c and c in sdf.columns:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # groupers
    g_all = sdf.groupby(name_col, dropna=False)
    if wait_col:
        valid_mask = sdf[wait_col].notna()
        g_wait = sdf[valid_mask].groupby(name_col, dropna=False)
    else:
        # if we couldn't find a wait column, treat none as valid (shift_worked=0, no avg)
        valid_mask = pd.Series(False, index=sdf.index)
        g_wait = g_all

    # build base from ALL rows (for non-YWT fields)
    agg = {name_col: "first"}
    if group_col: agg[group_col] = "last"
    if si_col:    agg[si_col]    = "sum"
    if nisr_col:  agg[nisr_col]  = "sum"
    if sbr_col:   agg[sbr_col]   = "sum"
    if att_col:   agg[att_col]   = "mean"
    if star_col:  agg[star_col]  = "max"

    base = g_all.agg(**{k: (k, v) for k, v in agg.items()}).reset_index(drop=True)

    # add wait metrics from VALID rows only
    if wait_col:
        avg_wait = g_wait[wait_col].mean()                     # NaNs ignored by mean
        shift_done = g_wait.size()                             # count only rows w/ YWT
        base = base.merge(avg_wait.rename("avg_wait"), left_on=name_col, right_index=True, how="left")
        base = base.merge(shift_done.rename("ShiftWorked"), left_on=name_col, right_index=True, how="left")
    else:
        base["avg_wait"] = pd.NA
        base["ShiftWorked"] = 0

    def fmt2(x): return "" if pd.isna(x) else f"{float(x):.2f}"

    cards = []
    for _, r in base.iterrows():
        raw_name = _upper(r.get(name_col, ""))
        disp_name = _display_name(raw_name)
        photo_uri = photos_lookup.get(_name_key(raw_name), "")
        cards.append({
            "name": disp_name,
            "group": (r.get(group_col) if group_col else "") or "-",
            "shift_worked": int(r["ShiftWorked"]) if pd.notna(r.get("ShiftWorked")) else 0,
            "avg_wait": fmt2(r.get("avg_wait")),
            "stars": "" if (star_col is None or pd.isna(r.get(star_col))) else int(r.get(star_col)),
            "attendance": "" if (att_col is None or pd.isna(r.get(att_col))) else f"{float(r.get(att_col)):.2f}",
            "si": 0 if (si_col is None or pd.isna(r.get(si_col))) else int(r.get(si_col)),
            "nisr": 0 if (nisr_col is None or pd.isna(r.get(nisr_col))) else int(r.get(nisr_col)),
            "sbr": 0 if (sbr_col is None or pd.isna(r.get(sbr_col))) else int(r.get(sbr_col)),
            "photo": photo_uri,
        })

    # Optional: if you want to HIDE people with 0 valid YWT shifts, uncomment:
    # cards = [c for c in cards if c["shift_worked"] > 0]

    return cards


# -------- safety overlay (link, not merge) with robust short-name matching --------
def _safety_lookup(selected_months: list[str]) -> dict[str, dict[str, int]]:
    safety_df = _safety_df_cached()
    if not isinstance(safety_df, pd.DataFrame) or safety_df.empty:
        return {}

    nm_c   = _col(safety_df, "Name", "NAME")
    si_c   = _col(safety_df, "SI")
    nisr_c = _col(safety_df, "NISR")
    sbr_c  = _col(safety_df, "SBR")
    m_c    = _col(safety_df, "Month", "MONTH")
    if not nm_c or not (si_c and nisr_c and sbr_c):
        return {}

    tmp = safety_df.copy()
    if selected_months and m_c:
        keep = [m[:3] for m in selected_months]
        tmp = tmp[tmp[m_c].astype(str).str[:3].isin(keep)]

    for c in (si_c, nisr_c, sbr_c):
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

    agg = tmp.groupby(nm_c, dropna=False)[[si_c, nisr_c, sbr_c]].sum().reset_index()

    exact = {}
    entries = []
    by_last = {}
    all_idxs = []

    def pack(r):
        return {
            "si":   int(r[si_c])   if pd.notna(r[si_c])   else 0,
            "nisr": int(r[nisr_c]) if pd.notna(r[nisr_c]) else 0,
            "sbr":  int(r[sbr_c])  if pd.notna(r[sbr_c])  else 0,
        }

    for _, r in agg.iterrows():
        key = _name_key(str(r[nm_c]))
        toks = set(key.split())
        first = (key.split() or [""])[0]
        last  = (key.split() or [""])[-1]
        metrics = pack(r)
        exact[key] = metrics
        idx = len(entries)
        entries.append((key, toks, first, last, metrics))
        by_last.setdefault(last, []).append(idx)
        all_idxs.append(idx)

    def best_candidate(q_key: str):
        if q_key in exact:
            return exact[q_key]

        q_toks = set(q_key.split())
        if not q_toks:
            return None
        q_first = (q_key.split() or [""])[0]
        q_last  = (q_key.split() or [""])[-1]

        pool = by_last.get(q_last, all_idxs) if q_last else all_idxs

        best_score, best_metrics = 0.0, None
        for i in pool:
            k, toks, first, last, metrics = entries[i]
            inter = len(q_toks & toks)
            union = len(q_toks | toks) or 1
            jacc  = inter / union
            subset_bonus = 0.35 if (q_toks.issubset(toks) or toks.issubset(q_toks)) else 0.0
            last_bonus   = 0.35 if (q_last and last and q_last == last) else 0.0
            first_init   = 0.10 if (q_first and first and q_first[0] == first[0]) else 0.0
            seq_weighted = 0.25 * difflib.SequenceMatcher(None, q_key, k).ratio()
            score = 0.55 * jacc + subset_bonus + last_bonus + first_init + seq_weighted
            if score > best_score:
                best_score, best_metrics = score, metrics

        return best_metrics if best_score >= 0.75 else None

    class _Proxy(dict):
        def get(self, k, default=None):
            v = super().get(k, None)
            if v is not None:
                return v
            return best_candidate(k) or default

    return _Proxy(exact)

# ========= helpers for robust base picking & debug =========
def _pick_base(tables: dict, tab: str):
    order_p123 = ["p123_enriched", "p123_ywt", "p123_long", "appended", "Combined", "combined"]
    order_p456 = ["p456_group_enriched", "p456_enriched", "appended", "Combined", "combined"]
    keys = order_p123 if tab.upper()=="P123" else order_p456
    return _first_df(tables, *keys), keys

def _debug_info(tables, df_before_filters, df_after_filters, photos_lookup, tab, selected_months):
    info = []
    info.append(f"[tab]={tab}  months={selected_months or 'ALL'}")
    info.append("[tables] " + ", ".join(f"{k}:{getattr(v,'shape',None)}" for k,v in tables.items()))
    info.append(f"[base_before_filters] shape={getattr(df_before_filters,'shape',None)}")
    info.append(f"[after_filters] shape={getattr(df_after_filters,'shape',None)}")
    hit_keys = set(photos_lookup.keys())
    if isinstance(df_after_filters, pd.DataFrame):
        name_col = _col(df_after_filters, "Name","NAME")
        if name_col:
            names = df_after_filters[name_col].astype(str).head(50).tolist()
            hits = sum(1 for n in names if _name_key(n) in hit_keys)
            info.append(f"[photos] first50 matched={hits}/{len(names)}")
    return "\n".join(info)

# =============================== Routes ===============================
def _resolve_tab():
    """
    Accept both ?tab=P123/P456 and ?only=P123/P456
    (so Main can link with either query name).
    Defaults to 'P123'.
    """
    raw = request.args.get("tab") or request.args.get("only") or "P123"
    raw = (raw or "").strip().upper()
    return "P456" if raw == "P456" else "P123"

@nonshift.get("/")
def index():
    # cache-buster & debug
    if request.args.get("reload") == "1":
        _CACHE.update({"ts": 0, "tables": None})
        _PHOTO_CACHE.update({"ts": 0, "photos_dict": {}})
        _SAFETY_CACHE.update({"ts": 0, "df": None})

    current_tab = _resolve_tab()
    selected_months = request.args.getlist("month")  # e.g. ['Sep','Oct']
    debug = request.args.get("debug") == "1"

    # ---- ADDED: prebuild month querystring for easy linking to /profile ----
    month_qs = "&".join(f"month={m}" for m in selected_months) if selected_months else ""

    tables = _tables_cached() or {}
    base, tried_keys = _pick_base(tables, current_tab)
    
    # if nothing selected, show a clear demo only then
    if base is None or base.empty:
        demo = pd.DataFrame([
            {"Name":"ABDUL HAKIM BIN YUSOF","group":"B*","Month":"Sep","SI":4,"NISR":2,"SBR":18,"STAR":1,"Attendance":98.2,"ywt.":10.12},
            {"Name":"AUSTIN CHUE","group":"D*","Month":"Sep","SI":4,"NISR":4,"SBR":20,"STAR":2,"Attendance":98.2,"ywt.":10.01},
            {"Name":"CALVIN HO","group":"A*","Month":"Sep","SI":3,"NISR":0,"SBR":16,"STAR":2,"Attendance":98.2,"ywt.":9.89},
        ])
        months_all = _order_months(demo["Month"].unique().tolist()) or MONTHS_ORDER
        cards = _make_cards(demo, {})
        safety_map = _safety_lookup(selected_months)
        if safety_map:
            for c in cards:
                m = safety_map.get(_name_key(c.get("name","")))
                if m:
                    c["si"], c["nisr"], c["sbr"] = m["si"], m["nisr"], m["sbr"]
        if debug:
            debug_text = "[FALLBACK] No base found. Tried keys: " + ", ".join(tried_keys)
            return render_template("nonshift.html", cards=cards, months=months_all,
                                   selected_months=selected_months, current_tab=current_tab,
                                   active_tab=current_tab, month_qs=month_qs,  # ADDED
                                   debug_text=debug_text)
        return render_template("nonshift.html", cards=cards, months=months_all,
                               selected_months=selected_months, current_tab=current_tab,
                               active_tab=current_tab, month_qs=month_qs)  # ADDED

    # normal path
    df0 = base.copy()
    df0 = _ensure_month_3(df0)
    df0 = _slice_tab(df0, current_tab)

    # filters (guard against filtering to empty -> relax)
    df = df0.copy()
    valid_groups = {"A*", "B*", "C*", "D*"}
    group_col = _col(df, "group", "Group", "GROUP")
    term_col  = _col(df, "terminal", "Terminal", "TERMINAL")

    if group_col:
        df = df[df[group_col].astype(str).str.strip().str.upper().isin(valid_groups)]
    if term_col:
        if df[term_col].astype(str).str.upper().isin([current_tab.upper()]).any():
            df = df[df[term_col].astype(str).str.strip().str.upper().eq(current_tab.upper())]

    if selected_months:
        keep = [m[:3] for m in selected_months]
        df = df[df["Month"].astype(str).str[:3].isin(keep)]

    if df.empty:
        df = df0.copy()
        if selected_months:
            keep = [m[:3] for m in selected_months]
            df = df[df["Month"].astype(str).str[:3].isin(keep)]

    months_all = _order_months(df["Month"].dropna().astype(str).str[:3].unique().tolist()) or MONTHS_ORDER
    photos_lookup = _photos_cached()
    cards = _make_cards(df, photos_lookup)
    
    # overlay safety
    safety_map = _safety_lookup(selected_months)
    if safety_map:
        for c in cards:
            m = safety_map.get(_name_key(c.get("name","")))
            if m:
                c["si"], c["nisr"], c["sbr"] = m["si"], m["nisr"], m["sbr"]

    if debug:
        dbg = _debug_info(tables, df0, df, photos_lookup, current_tab, selected_months)
        return render_template("nonshift.html", cards=cards, months=months_all,
                               selected_months=selected_months, current_tab=current_tab,
                               active_tab=current_tab, month_qs=month_qs,  # ADDED
                               debug_text=dbg)

    return render_template("nonshift.html",
        cards=cards,
        months=months_all,
        selected_months=selected_months,
        current_tab=current_tab,
        active_tab=current_tab,   # ADDED (alias for templates)
        month_qs=month_qs,        # ADDED (prebuilt ?month=... string)
    )

# --------- Convenience routes so you can link cleanly ---------
@nonshift.get("/p123")
def p123():
    return redirect(url_for(".index", tab="P123"))

@nonshift.get("/p456")
def p456():
    return redirect(url_for(".index", tab="P456"))
