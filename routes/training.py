# routes/training.py
import os
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


# ===================== ADMIN AUTH (simple) =====================

def is_admin() -> bool:
    """
    Choose ONE of these patterns (keep both checks for flexibility):
    - session["is_admin"] = True (set elsewhere in your app)
    - session["training_employee_role"] == "ADMIN" (if you later allow admin logins)
    """
    return bool(session.get("is_admin")) or session.get("training_employee_role") == "ADMIN"


def _get_booklet_config(role: str):
    """
    Returns (PAGES_DIR, booklet_endpoint, template, booklet_title)
    """
    role = (role or "").upper()

    if role == "YAS":
        from routes.yas_booklet import PAGES_DIR as YAS_PAGES_DIR
        return (YAS_PAGES_DIR, "yas_booklet.yas_page", "yas_booklet/page.html", "YAS Booklet")

    if role == "YSS":
        from routes.yss_booklet import PAGES_DIR as YSS_PAGES_DIR
        return (YSS_PAGES_DIR, "yss_booklet.yss_page", "yas_booklet/page.html", "YSS Booklet")

    abort(400, "Unknown booklet role")


def _max_page_from_dir(pages_dir: Path) -> int:
    if not pages_dir.exists():
        return 0
    pages = sorted(pages_dir.glob("*.html"))
    # if your pages are strictly 001..NNN, len() is fine
    return len(pages)


# ===================== LOGIN =====================

@training_bp.route("/training/login", methods=["POST"])
def training_login():
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "").strip().upper()

    if not name:
        abort(400, "Name is required")

    # ✅ allow ADMIN
    if role not in ("YAS", "YSS", "ADMIN"):
        abort(400, "Invalid role")

    # ✅ ADMIN path
    if role == "ADMIN":
        admin_pw = (request.form.get("admin_password") or "").strip()

        # Recommended: set this in environment variable
        expected = os.environ.get("TRAINING_ADMIN_PASSWORD", "wondercess")

        # If you haven't set env var yet, you can TEMPORARILY hardcode:
        # expected = "yourpassword"

        if not expected:
            abort(500, "TRAINING_ADMIN_PASSWORD not set on server")

        if admin_pw != expected:
            abort(403, "Invalid admin password")

        # Set admin session
        session["is_admin"] = True
        session["training_employee_name"] = name
        session["training_employee_role"] = "ADMIN"
        session.pop("training_employee_id", None)  # keep clean

        return redirect(url_for("training.admin_dashboard"))

    # ===== Trainee flow (YAS/YSS) =====
    db = get_db()
    cur = db.cursor()

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

    session["training_employee_id"] = row["id"]
    session["training_employee_name"] = name
    session["training_employee_role"] = role

    if role == "YAS":
        return redirect(url_for("yas_booklet.yas_page", page=1))

    if role == "YSS":
        return redirect(url_for("yss_booklet.yss_page", page=1))

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

def load_autosave_data(employee_id: int):
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


# ===================== PRINT ALL PAGES (trainee) =====================

@training_bp.route("/training/print")
def training_print():
    if "training_employee_id" not in session:
        abort(401)

    role = session.get("training_employee_role")
    pages_dir, booklet_endpoint, template, booklet_title = _get_booklet_config(role)

    employee_id = session["training_employee_id"]
    saved_data = load_autosave_data(employee_id)

    max_page = _max_page_from_dir(pages_dir)
    if max_page == 0:
        abort(404, "No booklet pages found.")

    pages_html = []
    for page in range(1, max_page + 1):
        page_code = f"{page:03d}"
        page_file = pages_dir / f"{page_code}.html"
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
            booklet_endpoint=booklet_endpoint,
            admin_view=False
        )
        pages_html.append(page_html)

    return render_template(
        "yas_booklet/print_all.html",
        pages_html=pages_html
    )


# ============================================================
# ===================== ADMIN (Step 2/3/4) ====================
# ============================================================

# ---------- Step 2A: Admin dashboard (list trainees) ----------
@training_bp.route("/training/admin")
def admin_dashboard():
    if not is_admin():
        abort(403)

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        SELECT e.id, e.name, e.role, MAX(tp.updated_at) AS last_updated
        FROM employees e
        LEFT JOIN training_progress tp ON e.id = tp.employee_id
        GROUP BY e.id
        ORDER BY (last_updated IS NULL), last_updated DESC, e.name ASC
    """)

    trainees = cur.fetchall()
    db.close()

    return render_template("admin/index.html", trainees=trainees)


# ---------- Step 2B: Admin trainee overview ----------
@training_bp.route("/training/admin/trainee/<int:employee_id>")
def admin_trainee_overview(employee_id):
    if not is_admin():
        abort(403)

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name, role FROM employees WHERE id = ?", (employee_id,))
    trainee = cur.fetchone()
    if not trainee:
        db.close()
        abort(404)

    cur.execute("""
        SELECT page_code, COUNT(*) AS fields_filled, MAX(updated_at) AS last_updated
        FROM training_progress
        WHERE employee_id = ?
        GROUP BY page_code
        ORDER BY page_code
    """, (employee_id,))
    pages = cur.fetchall()

    db.close()

    return render_template(
        "admin/trainee_overview.html",
        trainee=trainee,
        pages=pages
    )


# ---------- Step 3: Admin read-only booklet viewer ----------
@training_bp.route("/training/admin/booklet/<int:employee_id>/<int:page>")
def admin_view_page(employee_id, page):
    if not is_admin():
        abort(403)

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name, role FROM employees WHERE id = ?", (employee_id,))
    trainee = cur.fetchone()
    if not trainee:
        db.close()
        abort(404)

    pages_dir, booklet_endpoint, template, booklet_title = _get_booklet_config(trainee["role"])
    max_page = _max_page_from_dir(pages_dir)
    if max_page == 0:
        db.close()
        abort(404, "No booklet pages found.")

    if page < 1 or page > max_page:
        db.close()
        abort(404)

    page_code = f"{page:03d}"
    page_file = pages_dir / f"{page_code}.html"
    if not page_file.exists():
        db.close()
        abort(404)

    content_html = page_file.read_text(encoding="utf-8")

    cur.execute("""
        SELECT field_name, field_value
        FROM training_progress
        WHERE employee_id = ? AND page_code = ?
    """, (employee_id, page_code))
    saved_data = dict(cur.fetchall())

    db.close()

    return render_template(
        template,
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data,
        booklet_title=booklet_title,
        booklet_endpoint=booklet_endpoint,
        admin_view=True,
        trainee_name=trainee["name"],
        trainee_role=trainee["role"]
    )


# ---------- Step 4: Admin print full booklet for trainee ----------
@training_bp.route("/training/admin/print/<int:employee_id>")
def admin_print(employee_id):
    if not is_admin():
        abort(403)

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name, role FROM employees WHERE id = ?", (employee_id,))
    trainee = cur.fetchone()
    if not trainee:
        db.close()
        abort(404)

    pages_dir, booklet_endpoint, template, booklet_title = _get_booklet_config(trainee["role"])
    max_page = _max_page_from_dir(pages_dir)
    if max_page == 0:
        db.close()
        abort(404, "No booklet pages found.")

    db.close()

    saved_data = load_autosave_data(employee_id)

    pages_html = []
    for page in range(1, max_page + 1):
        page_code = f"{page:03d}"
        page_file = pages_dir / f"{page_code}.html"
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
            booklet_endpoint=booklet_endpoint,
            admin_view=True,
            trainee_name=trainee["name"],
            trainee_role=trainee["role"]
        )
        pages_html.append(page_html)

    return render_template(
        "yas_booklet/print_all.html",
        pages_html=pages_html
    )
