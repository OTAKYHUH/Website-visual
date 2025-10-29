# routes/profile.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from flask import (
    Blueprint,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

# ===== Your existing data helpers (unchanged) =====
from build.build_data import add_month_columns, get_tables

# Optional photos helper (same pattern as main/nonshift)
try:
    from build.photos import get_tables as get_photo_tables  # {"photos_dict": {...}}
except Exception:
    get_photo_tables = None

# Optional SI report index
try:
    from build.build_si_report import get_tables as get_si_tables  # expects {"SI Report": DataFrame}
except Exception:
    get_si_tables = None

profile = Blueprint("profile", __name__, url_prefix="/profile", template_folder="templates")

# ---------- caches ----------
_CACHE_TTL = 120
_CACHE = {"ts": 0.0, "tables": None}

_PHOTO_TTL = 300
_PHOTO_CACHE = {"ts": 0.0, "photos": {}}

_SI_TTL = 300
_SI_CACHE = {"ts": 0.0, "tables": {}}

# ===== Positive Contribution storage (XLSX with CSV fallback) =====
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PC_XLSX = DATA_DIR / "positive_contributions.xlsx"
PC_CSV = DATA_DIR / "positive_contributions.csv"
PC_COLUMNS = ["Name", "Description", "CreatedAt"]


def _pc_read_all() -> pd.DataFrame:
    # Prefer XLSX; fall back to CSV; else empty
    if PC_XLSX.exists():
        try:
            df = pd.read_excel(PC_XLSX, engine="openpyxl")
            for c in PC_COLUMNS:
                if c not in df.columns:
                    df[c] = "" if c != "CreatedAt" else pd.NaT
            return df[PC_COLUMNS]
        except Exception:
            pass
    if PC_CSV.exists():
        try:
            df = pd.read_csv(PC_CSV)
            if "CreatedAt" in df.columns:
                df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce")
            for c in PC_COLUMNS:
                if c not in df.columns:
                    df[c] = "" if c != "CreatedAt" else pd.NaT
            return df[PC_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=PC_COLUMNS)


def _pc_write(df: pd.DataFrame) -> None:
    # Try XLSX first (needs openpyxl). Retry a few times in case the file is locked.
    last_err: Optional[Exception] = None
    for _ in range(6):
        try:
            with pd.ExcelWriter(PC_XLSX, engine="openpyxl", mode="w") as w:
                df.to_excel(w, index=False)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.3)  # file lock backoff (Excel open?)
        except Exception as e:
            last_err = e
            break
    # Fallback to CSV so the save still succeeds
    try:
        df.to_csv(PC_CSV, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write XLSX ({last_err}); CSV fallback also failed: {e}") from e


def _pc_add(name: str, description: str) -> pd.DataFrame:
    name = str(name).strip()
    description = str(description).strip()
    now = pd.Timestamp.now(tz="Asia/Singapore")
    base = _pc_read_all()
    base = pd.concat(
        [base, pd.DataFrame([{"Name": name, "Description": description, "CreatedAt": now}])],
        ignore_index=True,
    )
    _pc_write(base)
    return base


def _pc_for_name(name: str) -> List[dict]:
    base = _pc_read_all()
    if base.empty:
        return []
    sub = base[base["Name"].astype(str) == str(name)]
    sub = sub.sort_values("CreatedAt", ascending=False)
    return sub[["Description"]].rename(columns={"Description": "Description"}).to_dict(orient="records")


# ========= helpers =========
PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "Staff Photo"
SI_DIR = Path(__file__).resolve().parents[1] / "static" / "SI"  # <— added: mirror photos path pattern
MONTHS_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
                photos = {_name_key(k): v for k, v in raw.items()}
            except Exception:
                photos = {}
        _PHOTO_CACHE.update({"ts": now, "photos": photos})
    return _PHOTO_CACHE["photos"]


def _si_tables():
    now = time.time()
    if (now - _SI_CACHE["ts"] > _SI_TTL) or not _SI_CACHE["tables"]:
        si = {}
        if callable(get_si_tables):
            try:
                # mirror photos: pass the folder path in
                si = get_si_tables(SI_DIR) or {}
            except Exception as e:
                # Show the exact failure in PythonAnywhere's web error log
                print(f"[SI] get_si_tables({SI_DIR}) failed: {type(e).__name__}: {e}")
                si = {}
        else:
            print("[SI] get_si_tables is None (import failed).")
        _SI_CACHE.update({"ts": now, "tables": si})
    return _SI_CACHE["tables"]


def _name_key(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.upper().replace("/", " ")
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _name_key_nospace(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.upper()
    t = re.sub(r"[^A-Z0-9]+", "", t)
    return t


def _display_name(n: str) -> str:
    if not n:
        return n
    return re.sub(r"\bD O\b", "D/O", n, flags=re.IGNORECASE)


def _find_col(df: pd.DataFrame, *cands: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        k = c.lower()
        if k in low:
            return low[k]
    return None


def _order_months(ms):
    s = {str(m)[:3] for m in ms if pd.notna(m)}
    return [m for m in MONTHS_ORDER if m in s] or MONTHS_ORDER


def _ensure_month_and_shiftdate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Add Month + Shift Date if missing
    if "Month" not in df.columns or "Shift Date" not in df.columns:
        date_col = _find_col(df, "EVENT_SHIFT_DT", "DATE", "Date")
        shift_col = _find_col(df, "EVENT_HR12_SHIFT_C", "SHIFT", "Shift")
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
    cands = [c for c in df.columns if any(k in c.lower() for k in ("ywt", "wait", "avg"))]
    for c in cands:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return cands[0] if cands else None


def _base_df_for_name(all_tables: Dict[str, pd.DataFrame], person_key: str) -> pd.DataFrame | None:
    for key in ("p123_enriched", "p456_group_enriched", "p456_enriched", "p123_ywt", "appended", "p123_long"):
        df = all_tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            nm = _find_col(df, "Name", "NAME")
            if not nm:
                continue
            sub = df[df[nm].astype(str).str.upper().str.replace("/", " ").str.contains(person_key, na=False)]
            if not sub.empty:
                return sub
    for key, df in all_tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty and ("Name" in df.columns or "NAME" in df.columns):
            nm = _find_col(df, "Name", "NAME")
            sub = df[df[nm].astype(str).str.upper().str.replace("/", " ").str.contains(person_key, na=False)]
            if not sub.empty:
                return sub
    return None


def _pick_full_base_df(all_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for key in ("p123_enriched", "p456_group_enriched", "p456_enriched", "p123_ywt", "appended", "p123_long"):
        df = all_tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    for _, df in all_tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty and (_find_col(df, "Name", "NAME") is not None):
            return df
    return pd.DataFrame()


def _which_plant(val: str) -> str | None:
    v = (val or "").upper().strip()
    if v in {"P123", "P456"}:
        return v
    if "456" in v:
        return "P456"
    if "123" in v:
        return "P123"
    return None


def _group_letter(g: str) -> str:
    if not isinstance(g, str):
        return ""
    s = g.upper()
    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\*", s)
    return m.group(1) if m else s.strip()


def _compute_group_avgs_for_shifts(
    df_all: pd.DataFrame, shifts: Iterable[str], selected_months: Optional[List[str]] = None
) -> List[Tuple[str, Optional[float]]]:
    if df_all.empty:
        return [(g, None) for g in "ABCDE"]

    df_all = _ensure_month_and_shiftdate(df_all)

    if selected_months:
        keep_m = {m[:3] for m in selected_months}
        df_all = df_all[df_all["Month"].astype(str).str[:3].isin(keep_m)]

    group_col = _find_col(df_all, "group", "Group", "GROUP")
    ywt_col = _guess_wait_col(df_all)
    if not group_col or not ywt_col or "Shift Date" not in df_all.columns:
        return [(g, None) for g in "ABCDE"]

    df_all = df_all.copy()
    df_all[ywt_col] = pd.to_numeric(df_all[ywt_col], errors="coerce")

    key_shifts = set([str(s).strip() for s in shifts if pd.notna(s) and str(s).strip()])
    sub = df_all[df_all["Shift Date"].astype(str).isin(key_shifts)].copy()
    if sub.empty:
        return [(g, None) for g in "ABCDE"]

    sub["_G"] = sub[group_col].map(_group_letter)
    grp_avg = sub.dropna(subset=[ywt_col]).groupby("_G")[ywt_col].mean()

    out: List[Tuple[str, Optional[float]]] = []
    for g in "ABCDE":
        val = grp_avg.get(g, float("nan"))
        out.append((g, None if pd.isna(val) else round(float(val), 2)))
    return out


@profile.get("/<path:raw_name>")
def person(raw_name: str):
    """
    Optional query params:
      - month=Jan&month=Feb
      - tab/cluster=P123|P456
      - date=01%20Sep%2025 & shift=D|N
    """
    selected_months = request.args.getlist("month")

    tab_hint = (request.args.get("cluster") or request.args.get("tab") or "").strip().upper()

    tables = _tables()
    person_key = _name_key(raw_name)
    df0 = _base_df_for_name(tables, person_key)
    if df0 is None or df0.empty:
        abort(404, description="Person not found in current tables.")

    df0 = _ensure_month_and_shiftdate(df0)

    nm = _find_col(df0, "Name", "NAME")
    if nm:
        df0 = df0[df0[nm].astype(str).str.upper().map(_name_key).eq(person_key)]

    plant_col = _find_col(df0, "Plant", "PLANT", "Source")
    if not tab_hint:
        inferred = None
        if plant_col and plant_col in df0.columns:
            for v in df0[plant_col].astype(str):
                p = _which_plant(v)
                if p:
                    inferred = p
                    break
        if not inferred:
            grp_col = _find_col(df0, "group", "Group", "GROUP")
            if grp_col and grp_col in df0.columns:
                for v in df0[grp_col].astype(str):
                    p = _which_plant(v)
                    if p:
                        inferred = p
                        break
        tab_hint = inferred or ""

    if tab_hint and plant_col and df0[plant_col].astype(str).str.upper().eq(tab_hint).any():
        df0 = df0[df0[plant_col].astype(str).str.upper().eq(tab_hint)]

    df = df0.copy()
    if selected_months:
        keep = [m[:3] for m in selected_months]
        df = df[df["Month"].astype(str).str[:3].isin(keep)]
        if df.empty:
            df = df0.copy()

    wait_col = _guess_wait_col(df)
    date_col = _find_col(df, "EVENT_SHIFT_DT", "DATE", "Date")

    _tmp_all = df.copy()
    if wait_col and wait_col in _tmp_all.columns:
        _tmp_all[wait_col] = pd.to_numeric(_tmp_all[wait_col], errors="coerce")
    if date_col and date_col in _tmp_all.columns:
        _tmp_all[date_col] = pd.to_datetime(_tmp_all[date_col], errors="coerce")

    if wait_col and date_col:
        _clean = _tmp_all[_tmp_all[wait_col].notna()].sort_values(date_col)
    else:
        _clean = _tmp_all.sort_values(date_col) if date_col else _tmp_all

    group_col = _find_col(df, "group", "Group", "GROUP")
    group_val = df[group_col].dropna().astype(str).iloc[-1] if group_col and not df.empty else ""
    shifts_done = int(len(df)) if not df.empty else 0
    months_all = _order_months(df0["Month"].dropna().astype(str).str[:3].unique().tolist())
    if wait_col and date_col:
        shifts_done = int(len(_clean))

    photos = _photos()
    photo_uri = photos.get(person_key, "")

    # chart series
    if not date_col:
        chart_x = list(range(1, len(df) + 1))
        chart_y = [float("nan")] * len(df)
    else:
        if _clean.empty or not wait_col:
            chart_x, chart_y = [], []
        else:
            grp = (
                _clean.groupby(_clean[date_col].dt.normalize(), dropna=False)[wait_col]
                .mean()
                .reset_index()
            )
            grp["DateLabel"] = pd.to_datetime(grp[date_col]).dt.strftime("%d %b %y")
            chart_x = grp["DateLabel"].tolist()
            chart_y = grp[wait_col].astype(float).tolist()

    # shift chips + SI support
    shift_dates: List[str] = []
    if date_col:
        sh_col = _find_col(df, "EVENT_HR12_SHIFT_C", "SHIFT", "Shift")
        if not _clean.empty:
            labels = pd.to_datetime(_clean[date_col]).dt.strftime("%d %b %y")
            if sh_col and sh_col in _clean.columns:
                shift_dates = (labels + " " + _clean[sh_col].astype(str)).tolist()
            else:
                shift_dates = labels.tolist()
    elif "Shift Date" in df.columns and wait_col:
        shift_dates = _clean.get("Shift Date", pd.Series([], dtype=str)).astype(str).tolist()

    disp_name = _display_name(raw_name.upper())
    qs_date = (request.args.get("date") or "").strip()
    qs_shift = (request.args.get("shift") or "").strip().upper()

    selected_full = f"{qs_date} {qs_shift or 'D'}" if qs_date else ""
    selected_date_str, selected_shift = "", "D"
    if selected_full:
        m = re.match(r"(\d{2}\s\w{3}\s\d{2,4})\s+([DN])", selected_full)
        if m:
            selected_date_str, selected_shift = m.group(1), m.group(2)

    # SI file links
    si_files: List[dict] = []
    try:
        si_bundle = _si_tables()
        si_df = si_bundle.get("SI Report")
        if isinstance(si_df, pd.DataFrame) and not si_df.empty:
            name_col = _find_col(si_df, "Name") or "Name"
            file_col = _find_col(si_df, "Filename") or "Filename"
            date_col_si = _find_col(si_df, "Shift Date") or "Shift Date"
            shift_col = _find_col(si_df, "Shift") or "Shift"

            if (name_col in si_df.columns) and (file_col in si_df.columns):
                me_key_nospace = _name_key_nospace(raw_name)
                # normalize both sides (match how names are stored for SI)
                sub = si_df[si_df[name_col].astype(str).map(_name_key_nospace) == me_key_nospace].copy()

                if date_col_si in sub.columns:
                    sub["_DT_"] = pd.to_datetime(sub[date_col_si].astype(str), format="%d%m%Y", errors="coerce")
                    sub["Month"] = sub["_DT_"].dt.strftime("%b")
                    sub["_LABEL"] = sub["_DT_"].dt.strftime("%d %b %y")
                    sub.rename(columns={date_col_si: "Date"}, inplace=True)
                else:
                    sub["_DT_"] = pd.NaT
                    sub["Month"] = ""
                    sub["_LABEL"] = ""
                    if "Date" not in sub.columns:
                        sub["Date"] = ""

                if shift_col in sub.columns:
                    sub["_SHIFT"] = sub[shift_col].astype(str).str.strip().str.upper().str[0].replace({"G": "N"})
                else:
                    sub["_SHIFT"] = ""

                sub["Shift Date"] = sub.apply(
                    lambda r: (f"{r['_LABEL']} {r['_SHIFT']}".strip() if pd.notna(r["_DT_"]) else ""), axis=1
                )

                if selected_months:
                    keep_m = [m[:3] for m in selected_months]
                    sub = sub[sub["Month"].isin(keep_m)]

                sub = sub.sort_values("_DT_", ascending=False, na_position="last")

                seen = set()
                for _, r in sub.iterrows():
                    fn = str(r[file_col]).strip()
                    if not fn or fn in seen:
                        continue
                    seen.add(fn)
                    si_files.append(
                        {
                            "filename": fn,
                            "href": url_for("static", filename=f"SI/{fn}"),
                            "date_label": ("" if pd.isna(r.get("_DT_")) else str(r.get("Shift Date", ""))),
                        }
                    )
    except Exception:
        si_files = []

    # Group chart numbers
    if qs_date:
        person_shifts = [f"{qs_date} {qs_shift or 'D'}"]
    else:
        if "Shift Date" in df.columns and not df["Shift Date"].empty:
            person_shifts = [str(s).strip() for s in df["Shift Date"] if pd.notna(s)]
        else:
            dcol = _find_col(df, "EVENT_SHIFT_DT", "DATE", "Date")
            scol = _find_col(df, "EVENT_HR12_SHIFT_C", "SHIFT", "Shift")
            if dcol:
                dts = pd.to_datetime(df[dcol], errors="coerce")
                mon = dts.dt.strftime("%b")
                sfx = (df[scol].astype(str) if scol else "")
                person_shifts = (dts.dt.strftime("%d ") + mon + dts.dt.strftime(" %y") + " " + sfx.astype(str)).tolist()
            else:
                person_shifts = []

    full_base = _pick_full_base_df(tables)
    group_ywt_rows = _compute_group_avgs_for_shifts(full_base, person_shifts, selected_months)

    group_chart_labels = [g for g, _ in group_ywt_rows]
    group_chart_ywt = [v if v is not None else None for _, v in group_ywt_rows]

    full_base_c = _ensure_month_and_shiftdate(full_base.copy())
    if selected_months and not full_base_c.empty:
        keep_m = {m[:3] for m in selected_months}
        full_base_c = full_base_c[full_base_c["Month"].astype(str).str[:3].isin(keep_m)]

    ywt_col_all = _guess_wait_col(full_base_c) if not full_base_c.empty else None
    grp_col_all = _find_col(full_base_c, "group", "Group", "GROUP") if not full_base_c.empty else None
    name_col_all = _find_col(full_base_c, "Name", "NAME") if not full_base_c.empty else None

    if (ywt_col_all and grp_col_all and ("Shift Date" in full_base_c.columns) and len(person_shifts) > 0):
        full_base_c = full_base_c.copy()
        full_base_c[ywt_col_all] = pd.to_numeric(full_base_c[ywt_col_all], errors="coerce")
        full_base_c["Shift Date"] = full_base_c["Shift Date"].astype(str)

        _sub = full_base_c[full_base_c["Shift Date"].isin([s for s in person_shifts if s])].copy()
        _sub["_G"] = _sub[grp_col_all].map(_group_letter)

        counts_shifts = _sub.dropna(subset=[ywt_col_all]).groupby("_G")["Shift Date"].nunique()

        if name_col_all and name_col_all in _sub.columns:
            counts_emps = _sub.dropna(subset=[ywt_col_all]).groupby("_G").size()
        else:
            counts_emps = pd.Series({})

        group_chart_counts = [int(counts_shifts.get(g, 0)) for g in group_chart_labels]
        group_chart_empcounts = [int(counts_emps.get(g, 0)) for g in group_chart_labels]
    else:
        group_chart_counts = [0 for _ in group_chart_labels]
        group_chart_empcounts = [0 for _ in group_chart_labels]

    use_emp_bars = bool(qs_date)

    # Positive Contribution list for this person
    positive_contribs = _pc_for_name(_display_name(raw_name.upper()))

    return render_template(
        "profile.html",
        person={
            "name": _display_name(raw_name.upper()),
            "group": group_val or "-",
            "photo": photo_uri,
            "shifts_done": shifts_done,
        },
        months=months_all,
        selected_months=selected_months,
        chart_x=chart_x,
        chart_y=[(None if (pd.isna(v) or v == "") else float(v)) for v in chart_y],
        shift_dates=shift_dates,
        tab_hint=tab_hint or "P123",
        y_min=(6 if (tab_hint or "P123") == "P456" else 8),
        si_files=si_files,
        group_ywt_rows=group_ywt_rows,
        group_chart_labels=group_chart_labels,
        group_chart_ywt=group_chart_ywt,
        group_chart_counts=group_chart_counts,
        group_chart_empcounts=group_chart_empcounts,
        selected_date_str=selected_date_str,
        selected_shift=selected_shift,
        use_emp_bars=use_emp_bars,
        positive_contribs=positive_contribs,  # NEW
    )


# convenience: /profile?name=...
@profile.get("/")
def by_query():
    name = request.args.get("name")
    if not name:
        return redirect(url_for("main.index"))
    return redirect(url_for(".person", raw_name=name))


# ===== API: add Positive Contribution (JSON: {name, description}) =====
@profile.post("/contrib")
def add_contribution():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", "")).strip()
        if not name or not description:
            return jsonify({"ok": False, "error": "Missing name/description"}), 400

        _pc_add(name, description)
        items = _pc_for_name(name)
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
