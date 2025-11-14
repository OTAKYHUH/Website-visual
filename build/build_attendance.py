# build_attendance.py — merge multiple sheets then tally from monthly columns only
from __future__ import annotations

import re, glob
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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


def _initials(s: str) -> str:
    s = _name_key(s)
    parts = [p for p in s.split() if p not in {"D", "O"}]
    return "".join(p[0] for p in parts if p)


def _to_int(v) -> int:
    try:
        return int(pd.to_numeric(v, errors="coerce") or 0)
    except Exception:
        return 0


def _find_files(explicit: Optional[str]) -> List[Path]:
    if explicit:
        p = Path(explicit)
        return [p] if p.exists() else []

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
        if f not in seen:
            uniq.append(f)
            seen.add(f)
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
                        out.append(c); break
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
    """
    merged: List[pd.DataFrame] = []
    used_names: List[str] = []

    # prefer cached formula values; though we don't read totals, it doesn't hurt
    try:
        xls = pd.ExcelFile(xl, engine="openpyxl", engine_kwargs={"data_only": True})
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
                    merged.append(df); used_names.append(nm); picked_idx.add(i)

    # 2) supplement with any other sheet that has month columns
    for i, nm in enumerate(names):
        if i in picked_idx:
            continue
        df = _parse_sheet(xls, nm)
        if df is not None and _df_has_month_cols(df):
            merged.append(df); used_names.append(nm); picked_idx.add(i)

    if not merged:
        return None, used_names

    # Normalize set union of columns; pandas will align by column names
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
            total_sum = float(sum(pd.to_numeric(df[c], errors="coerce").fillna(0).sum() for c in cols))
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
            if c == k or c.startswith(k + "_"):
                return c
    return None


# ---------- matching & extraction ----------
def _match_row(df: pd.DataFrame, name: Optional[str], employee_id: Optional[str], staff_no: Optional[str]) -> Optional[pd.Series]:
    """Pick the richest matching row in the merged DataFrame."""
    if df is None or df.empty:
        return None

    month_cols = _sheet_month_columns(df)

    def _score_row(r: pd.Series) -> int:
        return int(sum(_to_int(r.get(c, 0)) for c in month_cols))

    def _pick_best(sub: pd.DataFrame) -> Optional[pd.Series]:
        if sub.empty:
            return None
        best_i, best_s = None, -1
        for i, r in sub.iterrows():
            sc = _score_row(r)
            if sc > best_s:
                best_s, best_i = sc, i
        return sub.loc[best_i] if best_i is not None else None

    try:
        emp = _col(df, "Employee ID", "EMPLOYEE ID", "Emp ID", "EMP_ID")
        stf = _col(df, "Staff No", "STAFF NO", "StaffNo", "STAFF_NO")
        nam = _col(df, "Staff Name (JO)", "Staff Name", "Name", "NAME")

        if stf and staff_no:
            sub = df[df[stf].astype(str).str.upper().str.strip() == str(staff_no).upper().strip()]
            best = _pick_best(sub)
            if best is not None:
                return best

        if emp and employee_id:
            sub = df[df[emp].astype(str).str.upper().str.strip() == str(employee_id).upper().strip()]
            best = _pick_best(sub)
            if best is not None:
                return best

        if nam and name:
            key = _name_key(name)
            sub1 = df[df[nam].astype(str).map(_name_key) == key]
            best = _pick_best(sub1)
            if best is not None:
                return best

            nsp = _name_nospace(name)
            sub2 = df[df[nam].astype(str).map(_name_nospace) == nsp]
            best = _pick_best(sub2)
            if best is not None:
                return best

            ini = _initials(name)
            if ini:
                cand = df[df[nam].notna()].copy()
                cand["_INI_"] = cand[nam].astype(str).map(_initials)
                sub3 = cand[cand["_INI_"].str.contains("^" + re.escape(ini), na=False)]
                best = _pick_best(sub3)
                if best is not None:
                    return best
    except Exception:
        return None

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
    try:
        files = _find_files(path)
        df, picked, used_sheets = _pick_best_file(files)

        if df is None:
            return {"attendance_stats": _fallback_from_mc(month, name, employee_id, staff_no)}

        row = _match_row(df, name, employee_id, staff_no)
        stats = _stats(row, month)

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
        print(f"[ATT ERROR] {type(e).__name__} {e}")
        return {"attendance_stats": _fallback_from_mc(month, name, employee_id, staff_no)}
