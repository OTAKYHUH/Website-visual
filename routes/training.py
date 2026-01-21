# routes/training.py

import sqlite3
from pathlib import Path
from flask import (
    Blueprint, request, redirect, url_for,
    session, abort, render_template
)

training_bp = Blueprint("training", __name__)

# ===================== PATHS =====================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "training.db"

# ===================== DB =====================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ===================== LOGIN =====================

@training_bp.route("/training/login", methods=["POST"])
def training_login():
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "").strip().upper()

    # ---------- Validation ----------
    if not name:
        abort(400, "Name is required")

    if role not in ("YAS", "YSS"):
        abort(400, "Invalid role")

    db = get_db()
    cur = db.cursor()

    # ---------- Insert / fetch employee ----------
    cur.execute("""
        INSERT OR IGNORE INTO employees (name, role)
        VALUES (?, ?)
    """, (name, role))

    cur.execute("""
        SELECT id FROM employees
        WHERE name = ? AND role = ?
    """, (name, role))

    row = cur.fetchone()
    db.commit()
    db.close()

    if not row:
        abort(500, "Failed to login training user")

    # ---------- Session ----------
    session["training_employee_id"] = row["id"]
    session["training_employee_name"] = name
    session["training_employee_role"] = role

    # ---------- Role-based routing ----------
    if role == "YAS":
        return redirect(url_for("yas_booklet.yas_page", page=1))

    if role == "YSS":
        return redirect(url_for("yss_booklet.yss_page", page=1))

    # Fallback (should never hit)
    abort(400, "Unhandled role")

# ===================== AUTOSAVE =====================

@training_bp.route("/training/autosave", methods=["POST"])
def training_autosave():
    if "training_employee_id" not in session:
        return {"status": "unauthorized"}, 401

    data = request.get_json(force=True)

    page_code = data.get("page_code")
    field_name = data.get("field")
    field_value = data.get("value", "")

    if not page_code or not field_name:
        return {"status": "bad request"}, 400

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT INTO training_progress
            (employee_id, page_code, field_name, field_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(employee_id, page_code, field_name)
        DO UPDATE SET
            field_value = excluded.field_value,
            updated_at = CURRENT_TIMESTAMP
    """, (
        session["training_employee_id"],
        page_code,
        field_name,
        field_value
    ))

    db.commit()
    db.close()

    return {"status": "saved"}

# ===================== LOAD ALL SAVED DATA =====================

def load_autosave_data(employee_id):
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        SELECT page_code, field_name, field_value
        FROM training_progress
        WHERE employee_id = ?
    """, (employee_id,))

    data = {}
    for row in cur.fetchall():
        data.setdefault(row["page_code"], {})[row["field_name"]] = row["field_value"]

    db.close()
    return data

# ===================== PRINT ALL PAGES =====================

@training_bp.route("/training/print")
def training_print():
    if "training_employee_id" not in session:
        abort(401)

    role = session.get("training_employee_role")

    if role == "YAS":
        from routes.yas_booklet import PAGES_DIR, yas_page as page_endpoint
        booklet_endpoint = "yas_booklet.yas_page"
        template = "yas_booklet/page.html"
        booklet_title = "YAS Booklet"

    elif role == "YSS":
        from routes.yss_booklet import PAGES_DIR
        booklet_endpoint = "yss_booklet.yss_page"
        template = "yas_booklet/page.html"
        booklet_title = "YSS Booklet"

    else:
        abort(400)

    employee_id = session["training_employee_id"]
    saved_data = load_autosave_data(employee_id)

    pages = sorted(PAGES_DIR.glob("*.html"))
    max_page = len(pages)

    pages_html = []

    for page in range(1, max_page + 1):
        page_code = f"{page:03d}"
        page_file = PAGES_DIR / f"{page_code}.html"

        if not page_file.exists():
            continue

        content_html = page_file.read_text(encoding="utf-8")

        page_html = render_template(
            template,
            content_html=content_html,
            page=page,
            max_page=max_page,
            saved_data=saved_data.get(page_code, {}),
            booklet_title=booklet_title,
            booklet_endpoint=booklet_endpoint
        )

        pages_html.append(page_html)

    return render_template(
        "yas_booklet/print_all.html",
        pages_html=pages_html
    )
