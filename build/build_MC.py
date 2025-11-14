#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build/build_mc_details.py – robust, returns empty months if not found
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import re

# ====== CONFIG ======
BASE = Path(__file__).resolve().parents[1]
DEFAULT_PATH = BASE / "static" / "[NEW] YOD MASTER LIST (MC_UL_TURNOUT).xlsx"
DETAIL_SHEETS = ["MC Consolidation (P1-3)", "MC Consolidation (P4-6)"]

# We’ll accept header aliases and standardize them to these keys:
COL_EMP_ID   = "Employee ID"
COL_STAFF_NO = "Staff No"
COL_NAME     = "Staff Name"

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]

# cache like photos
_DF_CACHE: dict[str, dict] = {}

# ---------- helpers ----------
_ALIASES = {
    COL_EMP_ID:   { "Emp ID","EmployeeID","Emp_ID","ID" },
    COL_STAFF_NO: { "StaffNo","Staff_No","No","Staff Number" },
    COL_NAME:     { "Staff Name (JO)","Name","StaffName","Full Name" },
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=colmap)
    for target, alts in _ALIASES.items():
        if target not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: target})
                    break
    return df

def _load_mc_df(path: str | Path = DEFAULT_PATH,
                sheet_names: list[str] = DETAIL_SHEETS) -> pd.DataFrame:
    path = Path(path)
    mtime = path.stat().st_mtime
    k = str(path.resolve())
    cached = _DF_CACHE.get(k)
    if cached and abs(cached.get("ts", 0) - mtime) < 1:
        return cached["df"].copy()

    frames = []
    # try declared sheets; if missing, fall back to every sheet containing "MC Consolidation"
    xls = pd.ExcelFile(path)
    wanted = [s for s in sheet_names if s in xls.sheet_names]
    if not wanted:
        wanted = [s for s in xls.sheet_names if "MC Consolidation" in s]

    for sh in wanted:
        try:
            df = pd.read_excel(path, sheet_name=sh)
            df["__source_sheet__"] = sh
            frames.append(_standardize_columns(df))
        except Exception:
            pass

    if not frames:
        raise FileNotFoundError("No MC consolidation sheets found.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    _DF_CACHE[k] = {"ts": mtime, "df": df.copy()}
    return df

# ---- tolerant name matching (short forms, initials) ----
_SPACER_RE = re.compile(r"[^A-Z0-9]+")
def _norm(s: str) -> str:
    if s is None: return ""
    t = str(s).upper()
    t = (t.replace(" MOHD ", " MOHAMMAD ")
          .replace(" MHD ", " MOHAMMAD ")
          .replace(" MD ", " MOHAMMAD ")
          .replace(" MOHAMED ", " MOHAMMAD ")
          .replace(" MUHAMED ", " MOHAMMAD ")
          .replace(" MOHAMMAD ", " MUHAMMAD ")
          .replace(" NUR ", " NOR "))
    for token in (" BIN ", " BINTI ", " D/O ", " S/O "):
        t = t.replace(token, " ")
    return _SPACER_RE.sub(" ", t).strip()

def _initials(s: str) -> str:
    parts = [p for p in _norm(s).split() if p]
    return " ".join(p[0] for p in parts)

def _variants(s: str) -> set[str]:
    base = _norm(s)
    parts = base.split()
    vs = {base, _initials(base)}
    drop = {"BIN","BINTI","A/L","A/P","D/O","S/O"}
    vs.add(" ".join([p for p in parts if p not in drop]))
    if len(parts) >= 2:
        vs.add(f"{parts[0]} {parts[-1]}")
        vs.add(f"{parts[0][0]} {parts[-1]}")
        vs.add(f"{parts[0]} {' '.join(p[0] for p in parts[1:-1])} {parts[-1]}")
    return {v for v in vs if v}

def _name_match(cell: str, query: str) -> bool:
    if not cell or not query: return False
    c = _norm(cell); q = _norm(query)
    if c == q: return True
    qv = _variants(query)
    if c in qv or _initials(c) in qv: return True
    return all(tok in c for tok in q.split())

def _find_person_row(df: pd.DataFrame,
                     staff_no: str | None,
                     employee_id: str | None,
                     name: str | None):
    # return (row, found_bool)
    if staff_no and COL_STAFF_NO in df.columns:
        sub = df.loc[df[COL_STAFF_NO].astype(str) == str(staff_no)]
        if not sub.empty: return sub.iloc[0], True
    if employee_id and COL_EMP_ID in df.columns:
        sub = df.loc[df[COL_EMP_ID].astype(str) == str(employee_id)]
        if not sub.empty: return sub.iloc[0], True
    if name and COL_NAME in df.columns:
        nkey = _norm(name)
        sub = df.loc[df[COL_NAME].astype(str).map(_norm) == nkey]
        if not sub.empty: return sub.iloc[0], True
        hits = df.loc[df[COL_NAME].astype(str).map(lambda s: _name_match(s, name))]
        if not hits.empty: return hits.iloc[0], True
    return None, False

def _empty_details(month: str) -> dict:
    return {
        "person": {"employee_id": None, "staff_no": None, "name": None, "source_sheet": None},
        "month": month,
        "month_detail": "",
        "all_months": {m: "" for m in MONTH_COLS},
        "months": {m: [] for m in MONTH_COLS},
        "not_found": True,
    }

def get_person_mc_details(df: pd.DataFrame,
                          month: str,
                          staff_no: str | None = None,
                          employee_id: str | None = None,
                          name: str | None = None) -> dict:
    # normalize month label
    mlabel = "June" if month.strip().lower() == "jun" else month.strip().title()

    row, ok = _find_person_row(df, staff_no, employee_id, name)
    if not ok:
        return _empty_details(mlabel)

    def _split_lines(val) -> list[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)): return []
        s = str(val).strip()
        return [p.strip() for p in s.splitlines() if p.strip()] if s else []

    all_text = {}
    months_list = {}
    for col in MONTH_COLS:
        v = row.get(col, "")
        txt = "" if (pd.isna(v) if isinstance(v, float) else (v is None)) else str(v)
        all_text[col] = txt
        months_list[col] = _split_lines(v)

    month_detail = all_text.get(mlabel, "")

    return {
        "person": {
            "employee_id": row.get(COL_EMP_ID),
            "staff_no": row.get(COL_STAFF_NO),
            "name": row.get(COL_NAME),
            "source_sheet": row.get("__source_sheet__", None),
        },
        "month": mlabel,
        "month_detail": month_detail,
        "all_months": all_text,
        "months": months_list,
        "not_found": False,
    }

def get_tables(path: str | Path = DEFAULT_PATH,
               month: str = "Jan",
               staff_no: str | None = None,
               employee_id: str | None = None,
               name: str | None = None) -> dict[str, dict]:
    df = _load_mc_df(path, DETAIL_SHEETS)
    details = get_person_mc_details(df, month=month, staff_no=staff_no, employee_id=employee_id, name=name)
    return {"mc_details": details}
