# routes/training.py
import sqlite3
import uuid
from pathlib import Path
from flask import (
    Blueprint, request, redirect, url_for,
    session, abort, send_file, render_template
)
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

training_bp = Blueprint("training", __name__)

# ===================== DATABASE =====================

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "training.db"
PDF_DIR = Path(__file__).resolve().parents[1] / "pdf" / "generated"
PDF_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ===================== ENSURE TABLES =====================
# 🔹 SAFE: runs every time, does nothing if table already exists

def ensure_submission_table():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS training_submissions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          employee_id INTEGER,
          pdf_path TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()


# ===================== LOGIN =====================

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

    session["training_employee_id"] = row["id"]
    session["training_employee_name"] = name
    session["training_employee_role"] = role

    return redirect(url_for("yas_booklet.yas_page", page=1))


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


# ===================== LOAD AUTOSAVE DATA =====================
# 🔹 USED FOR PDF GENERATION

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
        page = row["page_code"]
        data.setdefault(page, {})[row["field_name"]] = row["field_value"]

    db.close()
    return data


# ===================== SUBMIT & GENERATE PDF =====================

@training_bp.route("/training/submit", methods=["POST"])
def training_submit():
    if "training_employee_id" not in session:
        return redirect(url_for("home.role_selection"))

    # 🔹 Ensure table exists (SAFE)
    ensure_submission_table()

    employee_id = session["training_employee_id"]
    saved_data = load_autosave_data(employee_id)

    html_pages = []

    for page in range(1, 33):
        page_code = f"{page:03d}"

        page_html = render_template(
            "yas_booklet/page.html",
            page=page,
            max_page=32,
            content_html=render_template(
                f"yas_booklet/{page_code}.html"
            ),
            saved_data=saved_data.get(page_code, {})
        )

        html_pages.append(page_html)

    final_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        @page {{
          size: A4;
          margin: 15mm;
        }}
        .page-break {{
          page-break-after: always;
        }}
      </style>
    </head>
    <body>
      {"<div class='page-break'></div>".join(html_pages)}
    </body>
    </html>
    """

    pdf_id = uuid.uuid4().hex
    pdf_path = PDF_DIR / f"yas_{employee_id}_{pdf_id}.pdf"

    if not WEASYPRINT_AVAILABLE:
        abort(501, "PDF generation not available in local development")

    HTML(
        string=final_html,
        base_url=str(Path(__file__).parents[1])
    ).write_pdf(pdf_path)

    db = get_db()
    db.execute("""
        INSERT INTO training_submissions (employee_id, pdf_path)
        VALUES (?, ?)
    """, (employee_id, str(pdf_path)))
    db.commit()
    db.close()

    return redirect(url_for("training.download_pdf", pdf_id=pdf_id))


# ===================== DOWNLOAD PDF =====================

@training_bp.route("/training/download/<pdf_id>")
def download_pdf(pdf_id):
    if "training_employee_id" not in session:
        abort(401)

    employee_id = session["training_employee_id"]

    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT pdf_path FROM training_submissions
        WHERE employee_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (employee_id,))

    row = cur.fetchone()
    db.close()

    if not row:
        abort(404)

    return send_file(
        row["pdf_path"],
        as_attachment=True,
        download_name="YAS_OJT_Record.pdf"
    )
