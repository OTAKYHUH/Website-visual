# main.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Blueprint, render_template, request

from build.build_data import get_tables  # adjust path if needed

# PHOTOS (data-URIs) from photos.py (same as NonShift)
try:
    from build.photos import get_tables as get_photo_tables  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    get_photo_tables = None  # type: ignore

# -----------------------------------------------------------------------------
# Blueprint
# -----------------------------------------------------------------------------
main = Blueprint("main", __name__, template_folder="templates")

# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------
_CACHE_TTL_SEC = 120
_CACHE: Dict[str, object] = {"ts": 0.0, "tables": None}

_PHOTO_TTL_SEC = 300
_PHOTO_CACHE: Dict[str, object] = {"ts": 0.0, "photos_dict": {}}

# Local photo directory (used by build/photos.py)
PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------
MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
def _order_months(ms):
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s]

def _tables_cached() -> Dict[str, pd.DataFrame]:
    """Return data tables with a short TTL cache."""
    now = time.time()
    if (not _CACHE["tables"]) or (now - float(_CACHE["ts"]) > _CACHE_TTL_SEC):
        _CACHE["tables"] = get_tables(show_errors=True)
        _CACHE["ts"] = now
    return _CACHE["tables"] or {}


def _photos_cached() -> Dict[str, str]:
    """
    Return {NORMALISED_KEY: data-uri} built from PHOTO_DIR via build/photos.py.
    Uses a lightweight in-memory TTL cache.
    """
    now = time.time()
    expired = (now - float(_PHOTO_CACHE["ts"])) > _PHOTO_TTL_SEC
    empty = not _PHOTO_CACHE.get("photos_dict")

    if expired or empty:
        photos_dict: Dict[str, str] = {}
        if os.path.isdir(PHOTO_DIR) and callable(get_photo_tables):  # type: ignore
            try:
                bundle = get_photo_tables(PHOTO_DIR)  # {"photos_df": df, "photos_dict": dict}
                raw = (bundle or {}).get("photos_dict", {}) or {}
                photos_dict = {_name_key(k): v for k, v in raw.items()}
            except Exception:
                photos_dict = {}

        _PHOTO_CACHE.update({"ts": now, "photos_dict": photos_dict})

    return _PHOTO_CACHE["photos_dict"] or {}


# -----------------------------------------------------------------------------
# String utilities
# -----------------------------------------------------------------------------
def _name_key(s) -> str:
    """Uppercase, '/'→space, remove non-alnum, collapse spaces."""
    t = "" if pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _norm_colname(s: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _display_name(n: str) -> str:
    """Minor prettification for names (D O → D/O)."""
    if not n:
        return n
    return re.sub(r"\bD O\b", "D/O", n, flags=re.IGNORECASE)


# -----------------------------------------------------------------------------
# Dataframe helpers
# -----------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Fuzzy column lookup by normalized name, with a few fallbacks."""
    index = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_colname(cand)
        if key in index:
            return index[key]
    # broader fallbacks
    for alt in ("group", "groupcode", "empgroup", "grouping", "grp", "cluster", "category"):
        if alt in index:
            return index[alt]
    return None


def _filter_to_abcd(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Keep only rows whose group is A/B/C/D or A* B* etc."""
    col = df[group_col].astype(str).str.strip().str.upper()
    uses_star = col.str.match(r"^[ABCD]\*$").any()
    if uses_star:
        mask = col.isin({"A*", "B*", "C*", "D*"})
    else:
        mask = col.str.match(r"^[ABCD]")
    filtered = df[mask]
    if filtered.empty and not col.empty:
        fallback = col.str.contains(r"^\s*[ABCD]\s*\*?\s*$", regex=True)
        filtered = df[fallback]
    return filtered


# -----------------------------------------------------------------------------
# Manual designation mapping (fuzzy)
# -----------------------------------------------------------------------------
_DESIG_FULL: Dict[str, str] = {
    "SOE": "Senior Operations Executive",
    "OE": "Operations Executive",
    "AM": "Assistant Manager",
}

# Your fuzzy name → short code (keys normalized once)
_DESIG_RAW: Dict[str, str] = {
    "Abdul Hakim": "SOE",
    "Adnan": "SOE",
    "Austin Chue": "SOE",
    "Calvin": "SOE",
    "Chng ming hao": "SOE",
    "haidar": "OE",
    "Jacinta": "OE",
    "Joe Chan": "OE",
    "Johnny": "SOE",
    "Lai LiHong": "SOE",
    "Lim Chun Sern": "SOE",
    "Mervin": "SOE",
    "FAUZI": "SOE",
    "jamalullail": "SOE",
    "Farzhani": "SOE",
    "Jian Hong": "SOE",
    "Toh Chee Chong": "AM",
    "Velson": "AM",
    "Wan zulhelmi": "SOE",
}
_DESIG_MAP: Dict[str, str] = {_name_key(k): v for k, v in _DESIG_RAW.items()}


def _designation_for(person_name: str) -> str:
    """
    Return full designation based on fuzzy manual list.
    We match by normalized substring (longest key wins).
    """
    norm = _name_key(person_name)
    best_code: str | None = None
    best_len = -1
    for key, code in _DESIG_MAP.items():
        if key in norm or norm in key:
            if len(key) > best_len:
                best_code = code
                best_len = len(key)
    return _DESIG_FULL.get(best_code or "", "")


# -----------------------------------------------------------------------------
# People builder (kept identical in behavior)
# -----------------------------------------------------------------------------
def _get_people() -> List[Dict[str, str]]:
    """
    Build a list of people with minimal fields used by the UI:
      {'name','group','designation','ywt','safety','hsl','fish','points','photo'}
    """
    tables = _tables_cached()
    frames: List[pd.DataFrame] = []
    for key in ("appended", "p123_enriched", "p456_group_enriched", "p456_enriched", "p123_long"):
        df = tables.get(key)  # type: ignore[index]
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)

    if not frames:
        return []

    base = pd.concat(frames, ignore_index=True)

    name_col = _find_col(base, "Name")
    role_col = _find_col(base, "Role")  # optional; we’ll prefer our manual mapping
    group_col = _find_col(base, "Group")

    if not name_col:
        return []

    base[name_col] = base[name_col].astype(str).str.strip()
    base = base[base[name_col].ne("")]

    if group_col:
        base = _filter_to_abcd(base, group_col)

    # Minimal dedup on name; keep first values for optional columns
    cols = [name_col]
    if role_col:
        cols.append(role_col)
    if group_col:
        cols.append(group_col)
    dedup = base[cols].drop_duplicates(subset=[name_col], keep="first")

    photos_lookup = _photos_cached()

    people: List[Dict[str, str]] = []
    for _, r in dedup.iterrows():
        raw_name = str(r[name_col]).strip()
        group_val = str(r.get(group_col, "") or "")
        manual_desig = _designation_for(raw_name)
        fallback_desig = str(r.get(role_col, "") or "")

        people.append(
            {
                "name": _display_name(raw_name),
                "group": group_val,
                # prefer manual full title; if missing, fall back to any 'Role' column
                "designation": manual_desig or fallback_desig,
                "ywt": "-",
                "safety": "-",
                "hsl": "-",
                "fish": "-",
                "points": "-",
                "photo": photos_lookup.get(_name_key(raw_name), ""),
            }
        )

    people.sort(key=lambda x: x["name"])
    return people


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@main.route("/main")
def index():
    """
    Main page:
      - Reads optional ?month=… filters (forward-compatible with dropdown)
      - Builds people list
      - Passes months (empty for now) + selected months through to the template
    """
    selected_months = request.args.getlist("month")
    tables = _tables_cached()  # reuse your cache
    months_seen = []
    for df in tables.values():
        if isinstance(df, pd.DataFrame) and not df.empty and "Month" in df.columns:
            months_seen.extend(df["Month"].astype(str).str[:3].unique().tolist())
    months = _order_months(pd.unique(pd.Series(months_seen))) or MONTHS_ORDER
    people = _get_people()

    return render_template(
        "main.html",
        months=months,
        selected_months=selected_months,
        people=people,
    )
