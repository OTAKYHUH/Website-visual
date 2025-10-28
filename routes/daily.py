# routes/daily.py
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import re
import pandas as pd

# Pull the prepared tables from your build step
from build.build_H import get_tables as get_h_tables  # type: ignore

daily = Blueprint("daily", __name__, url_prefix="/daily", template_folder="templates")

# ----------------------------- lightweight cache -----------------------------
_H_CACHE: Dict[str, Any] = {"ts": 0.0, "tables": None}
_CACHE_TTL = 90  # seconds


def _now_ts() -> float:
    from time import time
    return time()


def _get_h_tables() -> Dict[str, pd.DataFrame]:
    """Cached read of all tables built by build_H."""
    if (_now_ts() - _H_CACHE["ts"]) < _CACHE_TTL and _H_CACHE["tables"] is not None:
        return _H_CACHE["tables"]  # type: ignore
    tables = get_h_tables()
    _H_CACHE["tables"] = tables
    _H_CACHE["ts"] = _now_ts()
    return tables  # type: ignore


# --------------------------------- helpers -----------------------------------
def _parse_date_loose(s: str | None) -> Optional[datetime]:
    s = (s or "").strip()
    for fmt in ("%d %b %y", "%d %b %Y", "%-d %b %y", "%-d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True).to_pydatetime()
    except Exception:
        return None


def _norm_shift_letter(s: str | None) -> str:
    s = (s or "").strip().upper()
    if s == "1":
        return "D"
    if s in {"2", "G"}:
        return "N"
    return s if s in {"D", "N"} else "D"


def _split_shift_date_token(sd: str | None) -> Tuple[Optional[datetime], str]:
    """
    '02 Sep 25 D' → (datetime(2025-09-02), 'D').
    Accepts variants like '02 Sep 2025', '1/9/2025 N', etc.
    """
    sd = (sd or "").strip()
    m = re.match(r"^(.*?)(?:\s+([DN]))?$", sd, flags=re.I)
    if not m:
        return None, "D"
    date_part = (m.group(1) or "").strip()
    shift = _norm_shift_letter(m.group(2) or "")
    dt = _parse_date_loose(date_part)
    return dt, shift


def _norm_df_shift_date_tuple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce helper columns:
      _sd_date  (datetime.date)
      _sd_shift 'D' or 'N'
    Source preference:
      1) 'Shift Date' like '02 Sep 25 D'
      2) else 'Dates' / 'Date' / 'START_DT' (+ optional 'Shift')
    """
    g = df.copy()

    # Shift (normalize if present)
    if "Shift" in g.columns:
        g["_sd_shift"] = g["Shift"].astype(str).map(_norm_shift_letter)
    else:
        g["_sd_shift"] = "D"

    # Date source
    date_col = next(
        (c for c in ("Shift Date", "ShiftDate", "Dates", "Date", "START_DT", "START DT", "Start Date") if c in g.columns),
        None,
    )
    if date_col is None:
        g["_sd_date"] = pd.NaT
        return g

    if date_col in ("Shift Date", "ShiftDate"):
        date_only = g[date_col].astype(str).str.replace(r"\s+[DNdn]$", "", regex=True)
        g["_sd_date"] = pd.to_datetime(date_only, errors="coerce", dayfirst=True).dt.date
        missing = g["_sd_shift"].isna() | (g["_sd_shift"] == "")
        if missing.any():
            sh = g.loc[missing, date_col].astype(str).str.extract(r"([DNdn])$")[0].fillna("")
            g.loc[missing, "_sd_shift"] = sh.str.upper().map(_norm_shift_letter)
    else:
        g["_sd_date"] = pd.to_datetime(g[date_col], errors="coerce", dayfirst=True).dt.date

    return g


def _ensure_time_labels(series: pd.Series) -> pd.Series:
    """Map anything time-like to 'H:00' labels for the x-axis."""
    def _as_label(v) -> str:
        if pd.isna(v):
            return ""
        try:
            ts = pd.to_datetime(v, dayfirst=True)
            return f"{ts.hour}:00"
        except Exception:
            s = str(v).strip()
            try:
                ts = pd.to_datetime(s, dayfirst=True)
                return f"{ts.hour}:00"
            except Exception:
                m = re.search(r"\b(\d{1,2}):\d{2}\b", s)
                return (m.group(1) + ":00") if m else ""
    return series.apply(_as_label)


def _hours_for_shift(shift: str) -> List[str]:
    """Day: 07→19, Night: 19→07 (overnight)."""
    s = (shift or "").upper()
    if s == "N":
        seq = list(range(19, 24)) + list(range(0, 8))  # 19..23, 0..7
    else:
        seq = list(range(7, 20))                       # 7..19
    return [f"{h}:00" for h in seq]


# ------------------------------ YWT / HOURLY ---------------------------------
def _build_ywt_payload(shift_date_str: str) -> Dict[str, Any]:
    """Return 3 series (Mounting / Avg YWT / Offloading) overall and per terminal, with shift-aware hours."""
    tables = _get_h_tables()
    hourly: pd.DataFrame = tables.get("Hourly")  # type: ignore

    req_dt, req_shift = _split_shift_date_token(shift_date_str)
    labels = _hours_for_shift(req_shift or "D")

    def _blank() -> Dict[str, Any]:
        blank = [None] * len(labels)
        return {"hours": labels, "overview": {"mounting": blank, "avg": blank, "offload": blank}, "terminals": {}}

    if hourly is None or hourly.empty or req_dt is None:
        return _blank()

    g = _norm_df_shift_date_tuple(hourly)
    mask = (g["_sd_date"] == req_dt.date()) & (g["_sd_shift"] == (req_shift or "D"))
    sub = g.loc[mask].copy()
    if sub.empty:
        return _blank()

    time_col = next((c for c in ("Time", "Hour", "EventTime") if c in sub.columns), None)
    sub["time_lbl"] = _ensure_time_labels(sub[time_col]) if time_col else ""

    def pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
        low = {str(c).strip().lower(): c for c in df.columns}
        for c in cands:
            key = c.strip().lower()
            if key in low:
                return low[key]
        return None

    # tolerant, case-insensitive lookups
    time_col = pick(sub, "time", "hour", "eventtime", "event_time", "event hour")
    col_avg  = pick(sub,
                    "ywt", "avg", "average", "averagewaittime",
                    "avg wait time", "avg_ywt", "avg ywt")
    col_mount = pick(sub, "mounting", "eqmt", "mount")
    col_off   = pick(sub, "offloading", "eqof", "offload")

    def agg_series(col: Optional[str]) -> List[Optional[float]]:
        if not col or col not in sub.columns:
            return [None] * len(labels)
        s = sub.groupby("time_lbl", dropna=False)[col].mean().reindex(labels)
        return [round(v, 2) if pd.notna(v) else None for v in s.tolist()]

    overview = {"mounting": agg_series(col_mount), "avg": agg_series(col_avg), "offload": agg_series(col_off)}

    terminals: Dict[str, Dict[str, List[Optional[float]]]] = {}
    term_col = next((c for c in ("Terminal", "TERMINAL", "terminal") if c in sub.columns), None)
    if term_col:
        for term, chunk in sub.groupby(term_col):
            def ag(df: pd.DataFrame, col: Optional[str]) -> List[Optional[float]]:
                if not col or col not in df.columns:
                    return [None] * len(labels)
                s = df.groupby("time_lbl", dropna=False)[col].mean().reindex(labels)
                return [round(v, 2) if pd.notna(v) else None for v in s.tolist()]
            terminals[str(term)] = {
                "mounting": ag(chunk, col_mount),
                "avg": ag(chunk, col_avg),
                "offload": ag(chunk, col_off),
            }

    return {"hours": labels, "overview": overview, "terminals": terminals}


# ----------------------------------- HSL -------------------------------------
def _pick_hsl_log_frame(tables: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Prefer Log sheet for donut counts."""
    items = [(k, v, str(k).strip().lower()) for k, v in tables.items()
             if isinstance(v, pd.DataFrame) and not v.empty]
    for rx in (r"^log$", r"hsl\s*log", r"log_agg", r"hsl"):
        for _k, df, low in items:
            if re.search(rx, low, flags=re.I):
                return df
    return None


def _pick_hsl_details_frame(tables: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Prefer Details sheet for per-event rows."""
    items = [(k, v, str(k).strip().lower()) for k, v in tables.items()
             if isinstance(v, pd.DataFrame) and not v.empty]
    for rx in (r"^details$", r"hsl\s*details", r"details_raw", r"detail"):
        for _k, df, low in items:
            if re.search(rx, low, flags=re.I):
                return df
    # fallback to Log if Details is truly absent
    return _pick_hsl_log_frame(tables)


def _canon_term(x: Any) -> str:
    s = str(x or "").upper().strip()
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    if s in {"P1", "PPT1"}: return "PPT1"
    if s in {"P2", "PPT2"}: return "PPT2"
    if s in {"P3", "PPT3"}: return "PPT3"
    if s in {"P4", "PPT4"}: return "PPT4"
    if s in {"P56SE", "P5", "P6", "P56"}: return "P56SE"
    if s.startswith("PPT1"): return "PPT1"
    if s.startswith("PPT2"): return "PPT2"
    if s.startswith("PPT3"): return "PPT3"
    if s.startswith("PPT4"): return "PPT4"
    return s or "PPT1"


def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in cands:
        key = c.strip().lower()
        if key in low:
            return low[key]
    return None


def _is_pass(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in {"good", "pass", "ok", "success", "passed"}


def _norm_op(v: Any) -> str:
    s = str(v or "").lower()
    if "mount" in s: return "Mounting"
    if "off" in s or "dis" in s: return "Offload"
    return "Other"


def _allowed_for_cluster(cluster: str | None) -> Optional[set[str]]:
    c = (cluster or "").upper()
    if c == "P123":
        return {"PPT1", "PPT2", "PPT3"}
    if c == "P456":
        return {"PPT4", "P56SE"}
    return None


def _build_hsl_payload(shift_date_str: str, cluster: str | None) -> Dict[str, Any]:
    """
    Live HSL strictly for the requested Shift Date (e.g., '02 Sep 25 D').

    - Donuts (pass/fail/total per op) come from the Log table.
    - Event table rows come from the Details table with your exact column mapping.

    Result text is left as-is for the table.
    """
    tables = _get_h_tables()
    log_df = _pick_hsl_log_frame(tables)
    det_df = _pick_hsl_details_frame(tables)

    if (log_df is None or log_df.empty) and (det_df is None or det_df.empty):
        return {"terminals": {}}

    # parse token
    req_dt, req_shift = _split_shift_date_token(shift_date_str)
    if req_dt is None:
        return {"terminals": {}}
    req_shift = req_shift or "D"

    # ----- LOG PART: compute donuts (Good/Total → Pass/Fail) -----
    terms_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    if log_df is not None and not log_df.empty:
        g = _norm_df_shift_date_tuple(log_df).copy()
        g = g.loc[(g["_sd_date"] == req_dt.date()) & (g["_sd_shift"] == req_shift)].copy()
        if not g.empty:
            col_term  = _pick(g, "terminal", "ppt", "area", "station")
            col_type  = _pick(g, "gateopstype", "gate ops type", "gate_ops_type", "operation", "type", "optype", "action")

            col_m_good = _pick(g, "mounting good", "mount good", "eqmt good", "m_good", "good_m")
            col_m_tot  = _pick(g, "mounting total", "mount total", "eqmt total", "m_total", "total_m", "mounting")
            col_o_good = _pick(g, "offload good", "off loading good", "eqof good", "o_good", "good_o", "offloading good")
            col_o_tot  = _pick(g, "offload total", "off loading total", "eqof total", "o_total", "total_o", "offloading")

            col_good  = _pick(g, "good", "no. good", "count good", "pass")
            col_total = _pick(g, "total", "no. total", "count total")

            if col_term:
                g["__term__"] = g[col_term].map(_canon_term)
            else:
                g["__term__"] = "PPT1"

            if col_type:
                g["__op__"] = g[col_type].map(_norm_op)
            else:
                g["__op__"] = "Other"

            for tname, chunk in g.groupby("__term__"):
                def from_dedicated() -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
                    if (col_m_good or col_m_tot) or (col_o_good or col_o_tot):
                        mg = int(pd.to_numeric(chunk.get(col_m_good, 0), errors="coerce").fillna(0).sum()) if col_m_good else 0
                        mt = int(pd.to_numeric(chunk.get(col_m_tot,  0), errors="coerce").fillna(0).sum()) if col_m_tot  else 0
                        og = int(pd.to_numeric(chunk.get(col_o_good, 0), errors="coerce").fillna(0).sum()) if col_o_good else 0
                        ot = int(pd.to_numeric(chunk.get(col_o_tot,  0), errors="coerce").fillna(0).sum()) if col_o_tot  else 0
                        return ( (mg, max(mt-mg,0), mt), (og, max(ot-og,0), ot) )
                    return None

                def from_good_total_by_type() -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
                    if not (col_good and col_total and "__op__" in chunk.columns):
                        return None
                    def agg_for(oplab: str) -> Tuple[int,int,int]:
                        sub = chunk.loc[chunk["__op__"] == oplab]
                        good = int(pd.to_numeric(sub[col_good], errors="coerce").fillna(0).sum())
                        tot  = int(pd.to_numeric(sub[col_total], errors="coerce").fillna(0).sum())
                        return good, max(tot-good, 0), tot
                    return agg_for("Mounting"), agg_for("Offload")

                def from_row_level() -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
                    if "__op__" in chunk.columns:
                        def agg(oplab: str) -> Tuple[int,int,int]:
                            m = (chunk["__op__"] == oplab)
                            tot = int(m.sum())
                            return (0, tot, tot)
                        return agg("Mounting"), agg("Offload")
                    return ( (0,0,0), (0,0,0) )

                result = from_dedicated() or from_good_total_by_type() or from_row_level()
                (m_good, m_fail, m_tot), (o_good, o_fail, o_tot) = result
                terms_stats[tname] = {
                    "mount": {"pass": m_good, "fail": m_fail, "total": m_tot},
                    "off":   {"pass": o_good, "fail": o_fail, "total": o_tot},
                }

    # ----- DETAILS PART: build rows with your exact mapping -----
    terms_rows: Dict[str, List[Dict[str, str]]] = {}
    if det_df is not None and not det_df.empty:
        d = _norm_df_shift_date_tuple(det_df).copy()
        d = d.loc[(d["_sd_date"] == req_dt.date()) & (d["_sd_shift"] == req_shift)].copy()
        if not d.empty:
            # strict mapping to Details headers
            # type
            col_type  = _pick(d, "gateopstype")
            # start: prefer Start Time then Gate In/ Prev Trans
            col_start = _pick(d, "start time") or _pick(d, "gate in/ prev trans", "gate in", "prev trans")
            # end: prefer End time then Activity Date
            col_end   = _pick(d, "end time") or _pick(d, "activity date")
            # others
            col_haul  = _pick(d, "haulier company")
            col_svc   = _pick(d, "service time")
            col_ct    = _pick(d, "cntr type")
            col_loc   = _pick(d, "location")
            col_res   = _pick(d, "result", "gateopsresult", "status", "outcome")
            col_term  = _pick(d, "terminal", "ppt", "area", "station")

            if col_term:
                d["__term__"] = d[col_term].map(_canon_term)
            else:
                d["__term__"] = "PPT1"

            if col_start:
                d["__start_dt__"] = pd.to_datetime(d[col_start], errors="coerce", dayfirst=True)
                d = d.sort_values("__start_dt__", ascending=False)

            def getv(row, col):
                return row[col] if (col and col in row and pd.notna(row[col])) else ""

            for tname, chunk in d.groupby("__term__"):
                rows: List[Dict[str, str]] = []
                for _, r in chunk.iterrows():
                    rows.append({
                        "type":    str(getv(r, col_type)),
                        "start":   str(getv(r, col_start)),  # Gate In/ Prev Trans (or Start Time)
                        "end":     str(getv(r, col_end)),    # Activity Date (or End time)
                        "haulier": str(getv(r, col_haul)),   # Haulier Company
                        "svc":     str(getv(r, col_svc)),    # Service Time
                        "ctype":   str(getv(r, col_ct)),     # CNTR TYPE
                        "loc":     str(getv(r, col_loc)),    # Location
                        "result":  str(getv(r, col_res)),    # leave as-is
                    })
                terms_rows[tname] = rows

    # ----- combine donuts + rows per terminal, apply cluster filter -----
    allowed = _allowed_for_cluster(cluster)
    all_terms = sorted(set(list(terms_stats.keys()) + list(terms_rows.keys())))
    out_terms: Dict[str, Any] = {}
    for t in all_terms:
        if allowed and t not in allowed:
            continue
        m = terms_stats.get(t, {}).get("mount", {"pass": 0, "fail": 0, "total": 0})
        o = terms_stats.get(t, {}).get("off",   {"pass": 0, "fail": 0, "total": 0})
        total_all = int(m.get("total", 0)) + int(o.get("total", 0))
        pass_all = int(m.get("pass", 0)) + int(o.get("pass", 0))
        pass_rate = (float(pass_all) / float(total_all) * 100.0) if total_all else 0.0

        out_terms[t] = {
            "mount": m,
            "off":   o,
            "rows":  terms_rows.get(t, []),
            "pass_rate": round(pass_rate, 1),
        }

    return {"terminals": out_terms}


# ---------------------------------- routes -----------------------------------
@daily.get("/", endpoint="index")
def daily_index():
    """
    Accepts either:
      /daily?sd=02%20Sep%2025%20D
    or
      /daily?date=02%20Sep%2025&shift=D
    Optional:
      &name=<employee>
      &cluster=P123|P456  (limits HSL to PPT1/2/3 vs PPT4/P56SE)
    """
    sd_arg = (request.args.get("sd") or "").strip()
    date_arg = (request.args.get("date") or "").strip()
    shift = _norm_shift_letter(request.args.get("shift"))
    cluster = (request.args.get("cluster") or "").upper()

    if sd_arg:
        dt, sh = _split_shift_date_token(sd_arg)
        if dt is None:
            dt = datetime.now()
        if sh in {"D", "N"}:
            shift = sh
        center_dt = dt
    else:
        dt = _parse_date_loose(date_arg) or datetime.now()
        center_dt = dt
        if shift not in {"D", "N"}:
            shift = "D"

    sd_canonical = center_dt.strftime("%d %b %y") + f" {shift}"

    # chips around the selected date; keep same shift on the selected item
    def _make_shift_chips(center: datetime, sh: str, n_back=6, n_fwd=4) -> List[str]:
        chips: List[str] = []
        for i in range(-n_back, n_fwd + 1):
            day = center + timedelta(days=i)
            chip_shift = sh if i == 0 else ("N" if (day.toordinal() % 2) else "D")
            chips.append(day.strftime("%d %b %y") + f" {chip_shift}")
        return chips

    shift_dates = _make_shift_chips(center_dt, shift)

    # build both payloads
    payload = _build_ywt_payload(sd_canonical)
    payload["hsl"] = _build_hsl_payload(sd_canonical, cluster)
    payload["meta"] = "Daily · Hourly YWT"
    payload["employee_name"] = (request.args.get("name") or "").strip()
    payload["shift_date"] = sd_canonical

    return render_template(
        "daily.html",
        the_date_string=center_dt.strftime("%d %b %y"),
        the_shift=shift,
        shift_dates=shift_dates,
        payload=payload,
        name=payload["employee_name"],
    )


@daily.get("/data")
def data_api():
    """Optional JSON endpoint if you need async refresh."""
    sd = request.args.get("sd")
    if not sd:
        date_arg = request.args.get("date", "")
        shift = _norm_shift_letter(request.args.get("shift"))
        if date_arg:
            sd = (date_arg.strip() + " " + shift).strip()
        else:
            return jsonify({"error": "missing sd"})
    cluster = (request.args.get("cluster") or "").upper()
    out = _build_ywt_payload(sd)
    out["hsl"] = _build_hsl_payload(sd, cluster)
    return jsonify(out)
