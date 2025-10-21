from __future__ import annotations
from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
import random

daily = Blueprint("daily", __name__, url_prefix="/daily", template_folder="templates")

# ---------------- helpers ----------------
def _parse_date_str(s: str | None) -> datetime:
    if not s:
        return datetime.now()
    for fmt in ("%d %b %y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.now()

def _fmt_date_str(d: datetime) -> str:
    return d.strftime("%d %b %y")

def _make_shift_chips(center: datetime, shift: str, n_back=6, n_fwd=4) -> list[str]:
    chips = []
    for i in range(-n_back, n_fwd + 1):
        day = center + timedelta(days=i)
        sh = shift if i == 0 else ("N" if (day.day + i) % 2 else "D")  # demo alternation
        chips.append(f"{_fmt_date_str(day)} {sh}")
    return chips

def _dummy_series(n: int, base=10.0):
    return [round(base + random.uniform(-2.0, 2.0) + 0.15 * (i % 6), 2) for i in range(n)]

def _build_dummy_payload() -> dict:
    hours = ['7:00','8:00','9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00']
    def pack():
        return {"mounting": _dummy_series(len(hours)),
                "offload":  _dummy_series(len(hours)),
                "avg":      _dummy_series(len(hours))}
    mount = _dummy_series(len(hours), 11.0)
    off   = _dummy_series(len(hours), 10.5)
    avg   = [round((m + o) / 2 - 0.4, 2) for m, o in zip(mount, off)]
    return {
        "hours": hours,
        "overview": {"mounting": mount, "offload": off, "avg": avg},
        "ppt1": pack(),
        "ppt2": pack(),
        "ppt3": pack(),
        "manning": {
            "bc":   [["Adjusted Deployment","11"],["Available Manpower","11"],["Max Deployable Fleet","12"],["Previous Forecast","11"],["Today Demand","13"]],
            "ppt2": [["Adjusted Deployment","48/47+1 SAMSUNG | 11"],["Available Manpower","64/63+1 SAMSUNG"],
                     ["Max Deployable Fleet","49 | 11"],["Previous Forecast","51 | 13"],["Today Demand","51 | 13"]],
            "ppt3": [["Adjusted Deployment","58/57+1 DOOSAN"],["Available Manpower","55/54+1 DOOSAN"],
                     ["Max Deployable Fleet","58"],["Previous Forecast","57"],["Today Demand","57"]],
            "fm":   [["Adjusted Deployment","15+4 EDO"],["Available Manpower","19"],
                     ["Max Deployable Fleet","21 +1 TLOAN"],["Previous Forecast","24"],["Today Demand","19"]],
        },
    }

# ---------------- routes ----------------
@daily.get("/")
def index():
    # e.g. /daily?date=01%20Sep%2025&shift=D
    raw_date = request.args.get("date")
    shift = (request.args.get("shift") or "D").upper()
    if shift not in ("D", "N"):
        shift = "D"

    dt = _parse_date_str(raw_date)
    the_date_string = _fmt_date_str(dt)
    shift_dates = _make_shift_chips(dt, shift)

    payload = _build_dummy_payload()
    payload["title"] = f"{the_date_string} {shift}"
    payload["meta"] = "Daily · P123"

    return render_template(
        "daily.html",
        the_date_string=the_date_string,
        the_shift=shift,
        shift_dates=shift_dates,
        payload=payload,
    )

# Optional JSON endpoint (for future async loading)
@daily.get("/data")
def data_api():
    return jsonify(_build_dummy_payload())
