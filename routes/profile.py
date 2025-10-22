# profile.py
from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, abort
import time, os, re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Reuse your data utilities
from build.build_data import get_tables, add_month_columns

# Optional photos helper (same pattern as main/nonshift)
try:
    from build.photos import get_tables as get_photo_tables  # {"photos_dict": {...}}
except Exception:
    get_photo_tables = None

profile = Blueprint("profile", __name__, url_prefix="/profile", template_folder="templates")

# ---------- caches ----------
_CACHE_TTL = 120
_CACHE = {"ts": 0.0, "tables": None}

_PHOTO_TTL = 300
_PHOTO_CACHE = {"ts": 0.0, "photos": {}}

PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"
MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _tables():
    now = time.time()
    if (not _CACHE["tables"]) or (now - _CACHE["ts"] > _CACHE_TTL):
        _CACHE["tables"] = get_tables(show_errors=True)
        _CACHE["ts"] = now
    return _CACHE["tables"] or {}

def _photos():
    now = time.time()
    if (now - _PHOTO_CACHE["ts"] > _PHOTO_TTL) or not _PHOTO_CACHE["photos"]:
        photos = {}
        if os.path.isdir(PHOTO_DIR) and callable(get_photo_tables):
            try:
                bundle = get_photo_tables(PHOTO_DIR) or {}
                raw = bundle.get("photos_dict", {}) or {}
                photos = { _name_key(k): v for k, v in raw.items() }
            except Exception:
                photos = {}
        _PHOTO_CACHE.update({"ts": now, "photos": photos})
    return _PHOTO_CACHE["photos"]

def _name_key(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _display_name(n: str) -> str:
    if not n: return n
    return re.sub(r"\bD O\b", "D/O", n, flags=re.IGNORECASE)

def _find_col(df: pd.DataFrame, *cands: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        k = c.lower()
        if k in low: return low[k]
    return None

def _order_months(ms):
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s] or MONTHS_ORDER

def _ensure_month_and_shiftdate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Add Month + Shift Date if missing
    if "Month" not in df.columns or "Shift Date" not in df.columns:
        date_col  = _find_col(df, "EVENT_SHIFT_DT","DATE","Date")
        shift_col = _find_col(df, "EVENT_HR12_SHIFT_C","SHIFT","Shift")
        if date_col:
            df = add_month_columns(df, date_col)
        if ("Shift Date" not in df.columns) and date_col and shift_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            mon = dt.dt.strftime("%b")
            df["Shift Date"] = dt.dt.strftime("%d ") + mon + dt.dt.strftime(" %y") + " " + df[shift_col].astype(str)
    # Normalise Month to 3 letters
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(str).str[:3]
    return df

def _guess_wait_col(df: pd.DataFrame) -> str | None:
    # find numeric column that looks like YWT / wait / avg
    cands = [c for c in df.columns if any(k in c.lower() for k in ("ywt","wait","avg"))]
    for c in cands:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return cands[0] if cands else None

def _base_df_for_name(all_tables: Dict[str,pd.DataFrame], person_key: str) -> pd.DataFrame | None:
    # Prefer enriched tables
    for key in ("p123_enriched","p456_group_enriched","p456_enriched","p123_ywt","appended","p123_long"):
        df = all_tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            nm = _find_col(df, "Name","NAME")
            if not nm: continue
            sub = df[df[nm].astype(str).str.upper().str.replace("/", " ").str.contains(person_key, na=False)]
            if not sub.empty:
                return sub
    # fallback: try any table with Name
    for key, df in all_tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty and ("Name" in df.columns or "NAME" in df.columns):
            nm = _find_col(df, "Name","NAME")
            sub = df[df[nm].astype(str).str.upper().str.replace("/", " ").str.contains(person_key, na=False)]
            if not sub.empty:
                return sub
    return None

# ======== NEW: soft plant/cluster inference helper (P123/P456) ========
def _which_plant(val: str) -> str | None:
    v = (val or "").upper().strip()
    if v in {"P123", "P456"}: return v
    if "456" in v: return "P456"
    if "123" in v: return "P123"
    return None
# ======================================================================

@profile.get("/<path:raw_name>")
def person(raw_name: str):
    """
    Example: /profile/MUHAMMAD JAMALULLAIL BIN MAHADI
    Optional query params:
      - month=Jan&month=Feb (multi)
      - tab=P123 or P456 (to hint plant filter if available)
      - cluster=P123 or P456 (alias of tab; will be honored if present)
    """
    selected_months = request.args.getlist("month")

    # --------- CHANGED: also accept ?cluster= in addition to ?tab= ----------
    tab_hint = (
        request.args.get("cluster")
        or request.args.get("tab")
        or ""
    ).strip().upper()
    # -----------------------------------------------------------------------

    tables = _tables()
    person_key = _name_key(raw_name)
    df0 = _base_df_for_name(tables, person_key)
    if df0 is None or df0.empty:
        abort(404, description="Person not found in current tables.")

    df0 = _ensure_month_and_shiftdate(df0)

    # Try to keep only this name precisely (best effort)
    nm = _find_col(df0, "Name","NAME")
    if nm:
        df0 = df0[df0[nm].astype(str).str.upper().map(_name_key).eq(person_key)]

    # ======== NEW: infer cluster if not explicitly provided =========
    plant_col = _find_col(df0, "Plant","PLANT","Source")
    if not tab_hint:
        inferred = None
        # 1) Try plant/source column
        if plant_col and plant_col in df0.columns:
            for v in df0[plant_col].astype(str):
                p = _which_plant(v)
                if p:
                    inferred = p
                    break
        # 2) Try group column (sometimes carries hints)
        if not inferred:
            grp_col = _find_col(df0, "group","Group","GROUP")
            if grp_col and grp_col in df0.columns:
                for v in df0[grp_col].astype(str):
                    p = _which_plant(v)
                    if p:
                        inferred = p
                        break
        # 3) Keep as "" if nothing found; downstream we still fallback to P123 at render
        tab_hint = inferred or ""
    # =================================================================

    # Optional plant/tab slice (honors explicit/inferred tab_hint)
    if tab_hint and plant_col and df0[plant_col].astype(str).str.upper().eq(tab_hint).any():
        df0 = df0[df0[plant_col].astype(str).str.upper().eq(tab_hint)]

    # Month filter
    df = df0.copy()
    if selected_months:
        keep = [m[:3] for m in selected_months]
        df = df[df["Month"].astype(str).str[:3].isin(keep)]
        if df.empty:  # relax to df0 if too tight
            df = df0.copy()
    # --- NEW: derive a clean, sorted view that excludes blank YWTs ---
    wait_col = _guess_wait_col(df)
    date_col = _find_col(df, "EVENT_SHIFT_DT", "DATE", "Date")

    # Copy + coerce types
    _tmp_all = df.copy()
    if wait_col and wait_col in _tmp_all.columns:
        _tmp_all[wait_col] = pd.to_numeric(_tmp_all[wait_col], errors="coerce")
    if date_col and date_col in _tmp_all.columns:
        _tmp_all[date_col] = pd.to_datetime(_tmp_all[date_col], errors="coerce")

    # Keep only rows that have a YWT value and sort by date ascending
    if wait_col and date_col:
        _clean = _tmp_all[_tmp_all[wait_col].notna()].sort_values(date_col)
    else:
        _clean = _tmp_all.sort_values(date_col) if date_col else _tmp_all

    # Summary bits
    group_col = _find_col(df, "group","Group","GROUP")
    group_val = (df[group_col].dropna().astype(str).iloc[-1] if group_col and not df.empty else "")
    shifts_done = int(len(df)) if not df.empty else 0
    months_all = _order_months(df0["Month"].dropna().astype(str).str[:3].unique().tolist())
    if wait_col and date_col:
        shifts_done = int(len(_clean))

    # Photo
    photos = _photos()
    photo_uri = photos.get(person_key, "")

    # Chart series (by date)
    wait_col = _guess_wait_col(df)
    date_col = _find_col(df, "EVENT_SHIFT_DT","DATE","Date")
    if not date_col:
        # no date? fake x with 1..N
        chart_x = list(range(1, len(df)+1))
        chart_y = [float("nan")] * len(df)
    else:
        if _clean.empty or not wait_col:
            chart_x, chart_y = [], []
        else:
            # Group by calendar date (normalized), take mean YWT per date
            grp = (_clean
                .groupby(_clean[date_col].dt.normalize(), dropna=False)[wait_col]
                .mean()
                .reset_index())
            # Format X as 'dd mmm yy'
            grp["DateLabel"] = pd.to_datetime(grp[date_col]).dt.strftime("%d %b %y")
            chart_x = grp["DateLabel"].tolist()
            chart_y = grp[wait_col].astype(float).tolist()

    # Shift-date strip
    shift_dates = []
    if date_col:
        sh_col = _find_col(df, "EVENT_HR12_SHIFT_C", "SHIFT", "Shift")
        if not _clean.empty:
            labels = pd.to_datetime(_clean[date_col]).dt.strftime("%d %b %y")
            if sh_col and sh_col in _clean.columns:
                shift_dates = (labels + " " + _clean[sh_col].astype(str)).tolist()
            else:
                shift_dates = labels.tolist()
    elif "Shift Date" in df.columns and wait_col:
        # Fallback: keep only rows with non-blank YWT and sort if possible
        shift_dates = _clean.get("Shift Date", pd.Series([], dtype=str)).astype(str).tolist()
    # Pretty name
    disp_name = _display_name(raw_name.upper())
    qs_date  = (request.args.get("date") or "").strip()
    qs_shift = (request.args.get("shift") or "").strip().upper()

    # 2) default selection = last chip if not provided
    selected_full = None
    if qs_date:
        selected_full = f"{qs_date} {qs_shift or 'D'}"
    elif shift_dates:
        selected_full = shift_dates[-1]  # pick the most recent pill by default (adjust if you prefer first)

    # 3) split into date + shift for linking to /daily
    selected_date_str, selected_shift = "", "D"
    if selected_full:
        m = re.match(r"(\d{2}\s\w{3}\s\d{2,4})\s+([DN])", selected_full)
        if m:
            selected_date_str, selected_shift = m.group(1), m.group(2)
    return render_template(
        "profile.html",
        person={
            "name": disp_name,
            "group": group_val or "-",
            "photo": photo_uri,
            "shifts_done": shifts_done,
        },
        months=months_all,
        selected_months=selected_months,
        chart_x=chart_x,
        chart_y=[(None if (pd.isna(v) or v=="") else float(v)) for v in chart_y],
        shift_dates=shift_dates,
        # If the template needs a default, it can still do "| default('P123')"
        # but we now pass the best-available explicit/inferred hint here.
        tab_hint=tab_hint or "P123",
    )

# convenience: /profile?name=...
@profile.get("/")
def by_query():
    name = request.args.get("name")
    if not name:
        return redirect(url_for("main.index"))
    return redirect(url_for(".person", raw_name=name))
