# routes/training.py
import sqlite3
from pathlib import Path
from flask import Blueprint, request, redirect, url_for, session, abort

training_bp = Blueprint("training", __name__)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "training.db"


def get_db():
    return sqlite3.connect(DB_PATH)


@training_bp.route("/training/login", methods=["POST"])
def training_login():
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "").strip()

    if not name or not role:
        abort(400, "Name and role are required")

    db = get_db()
    cur = db.cursor()

    # Create employee if not exists
    cur.execute("""
        INSERT OR IGNORE INTO employees (name, role)
        VALUES (?, ?)
    """, (name, role))

    # Fetch employee ID
    cur.execute("""
        SELECT id FROM employees
        WHERE name = ? AND role = ?
    """, (name, role))

    row = cur.fetchone()
    db.commit()
    db.close()

    if not row:
        abort(500, "Failed to fetch employee")

    # Store in session
    session["training_employee_id"] = row[0]
    session["training_employee_name"] = name
    session["training_employee_role"] = role

    return redirect(url_for("yas_booklet.yas_page", page=1))

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
        INSERT INTO training_progress (employee_id, page_code, field_name, field_value)
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
