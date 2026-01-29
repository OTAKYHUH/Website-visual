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


# ===================== SCHEMA (SUBMIT / LOCK) =====================

def ensure_schema():
    """
    Adds 'submitted' and 'submitted_at' columns to employees table if missing.
    Safe to call multiple times.
    """
    db = get_db()
    cur = db.cursor()

    cur.execute("PRAGMA table_info(employees)")
    cols = {r["name"] for r in cur.fetchall()}

    if "submitted" not in cols:
        cur.execute("ALTER TABLE employees ADD COLUMN submitted INTEGER NOT NULL DEFAULT 0")
    if "submitted_at" not in cols:
        cur.execute("ALTER TABLE employees ADD COLUMN submitted_at TEXT")

    db.commit()
    db.close()


def is_submitted(employee_id: int) -> bool:
    ensure_schema()
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT submitted FROM employees WHERE id = ?", (employee_id,))
    row = cur.fetchone()
    db.close()
    return bool(row and row["submitted"] == 1)


# ===================== ROLE HELPERS =====================

def is_admin() -> bool:
    return bool(session.get("is_admin")) or session.get("training_employee_role") == "ADMIN"


def is_mentor() -> bool:
    return bool(session.get("is_mentor")) or session.get("training_employee_role") == "MENTOR"


def require_admin_or_mentor():
    if not (is_admin() or is_mentor()):
        abort(403)


# ===================== EDIT / LOCK RULES =====================
# Page codes like "029", "031"

# ✅ Mentors can EDIT only these pages
MENTOR_EDIT_PAGES = {
    "YAS": {"001", "029", "030", "031", "032"},  # mentor can edit these YAS pages
    "YSS": {"001", "019", "020", "021", "022"},  # mentor can edit these YSS pages
}

# ✅ Trainees are read-only on these pages (even before submit)
TRAINEE_LOCK_PAGES = {
    "YAS": {"029", "030", "031", "032"},
    "YSS": {"019", "020", "021", "022"},
}


def mentor_can_edit(trainee_role: str, page_code: str) -> bool:
    trainee_role = (trainee_role or "").upper()
    return page_code in MENTOR_EDIT_PAGES.get(trainee_role, set())


def trainee_page_locked(trainee_role: str, page_code: str) -> bool:
    trainee_role = (trainee_role or "").upper()
    return page_code in TRAINEE_LOCK_PAGES.get(trainee_role, set())


# ===================== BOOKLET CONFIG =====================

def _get_booklet_config(role: str):
    role = (role or "").upper()

    if role == "YAS":
        from routes.yas_booklet import PAGES_DIR as YAS_PAGES_DIR
        return (YAS_PAGES_DIR, "yas_booklet.yas_page", "yas_booklet/page.html", "YAS Booklet")

    if role == "YSS":
        from routes.yss_booklet import PAGES_DIR as YSS_PAGES_DIR
        # (template reused)
        return (YSS_PAGES_DIR, "yss_booklet.yss_page", "yas_booklet/page.html", "YSS Booklet")

    abort(400, "Unknown booklet role")


def _max_page_from_dir(pages_dir: Path) -> int:
    if not pages_dir.exists():
        return 0
    pages = sorted(pages_dir.glob("*.html"))
    return len(pages)


def _build_nav_context(
    *,
    admin_view: bool,
    mentor_view: bool,
    page: int,
    max_page: int,
    trainee_role: str,
    employee_id: int | None = None,
):
    """
    Builds next/prev/jump URLs.
    - admin_view: uses admin endpoints
    - mentor_view: uses mentor endpoints
    - trainee: uses booklet endpoints
    """
    if admin_view:
        if employee_id is None:
            abort(500, "employee_id required for admin navigation")

        nav_endpoint = "training.admin_view_page"

        nav_prev_url = url_for(nav_endpoint, employee_id=employee_id, page=page - 1) if page > 1 else None
        nav_next_url = url_for(nav_endpoint, employee_id=employee_id, page=page + 1) if page < max_page else None
        nav_jump_base_url = url_for(nav_endpoint, employee_id=employee_id, page=1).rsplit("/", 1)[0]
        return {"nav_prev_url": nav_prev_url, "nav_next_url": nav_next_url, "nav_jump_base_url": nav_jump_base_url}

    if mentor_view:
        if employee_id is None:
            abort(500, "employee_id required for mentor navigation")

        nav_endpoint = "training.mentor_view_page"

        nav_prev_url = url_for(nav_endpoint, employee_id=employee_id, page=page - 1) if page > 1 else None
        nav_next_url = url_for(nav_endpoint, employee_id=employee_id, page=page + 1) if page < max_page else None
        nav_jump_base_url = url_for(nav_endpoint, employee_id=employee_id, page=1).rsplit("/", 1)[0]
        return {"nav_prev_url": nav_prev_url, "nav_next_url": nav_next_url, "nav_jump_base_url": nav_jump_base_url}

    _, trainee_endpoint, _, _ = _get_booklet_config(trainee_role)
    nav_prev_url = url_for(trainee_endpoint, page=page - 1) if page > 1 else None
    nav_next_url = url_for(trainee_endpoint, page=page + 1) if page < max_page else None
    nav_jump_base_url = url_for(trainee_endpoint, page=1).rsplit("/", 1)[0]
    return {"nav_prev_url": nav_prev_url, "nav_next_url": nav_next_url, "nav_jump_base_url": nav_jump_base_url}


# ===================== LOGIN =====================

@training_bp.route("/training/login", methods=["POST"])
def training_login():
    ensure_schema()

    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "").strip().upper()

    if not name:
        abort(400, "Name is required")

    if role not in ("YAS", "YSS", "ADMIN", "MENTOR"):
        abort(400, "Invalid role")

    # ✅ ADMIN path
    if role == "ADMIN":
        admin_pw = (request.form.get("admin_password") or "").strip()
        expected = os.environ.get("TRAINING_ADMIN_PASSWORD", "wondercess")

        if not expected:
            abort(500, "TRAINING_ADMIN_PASSWORD not set on server")
        if admin_pw != expected:
            abort(403, "Invalid admin password")

        session["is_admin"] = True
        session.pop("is_mentor", None)

        session["training_employee_name"] = name
        session["training_employee_role"] = "ADMIN"
        session.pop("training_employee_id", None)

        return redirect(url_for("training.admin_dashboard"))

    # ✅ MENTOR path
    if role == "MENTOR":
        mentor_pw = (request.form.get("mentor_password") or "").strip()
        expected = os.environ.get("TRAINING_MENTOR_PASSWORD", "mentorcess")

        if not expected:
            abort(500, "TRAINING_MENTOR_PASSWORD not set on server")
        if mentor_pw != expected:
            abort(403, "Invalid mentor password")

        session["is_mentor"] = True
        session.pop("is_admin", None)

        session["training_employee_name"] = name
        session["training_employee_role"] = "MENTOR"
        session.pop("training_employee_id", None)

        return redirect(url_for("training.mentor_dashboard"))

    # ===== Trainee flow (YAS/YSS) =====
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO employees (name, role)
        VALUES (?, ?)
    """, (name, role))

    cur.execute("""
        SELECT id, submitted
        FROM employees
        WHERE name = ? AND role = ?
    """, (name, role))

    row = cur.fetchone()
    db.commit()
    db.close()

    if not row:
        abort(500, "Failed to login training user")

    session.pop("is_admin", None)
    session.pop("is_mentor", None)

    session["training_employee_id"] = row["id"]
    session["training_employee_name"] = name
    session["training_employee_role"] = role

    # ✅ If already submitted, redirect to confirmation instead of booklet (trainee only)
    if row["submitted"] == 1:
        return redirect(url_for("training.confirmation"))

    if role == "YAS":
        return redirect(url_for("yas_booklet.yas_page", page=1))

    if role == "YSS":
        return redirect(url_for("yss_booklet.yss_page", page=1))

    abort(400, "Unhandled role")


# ===================== SUBMIT / LOCK (TRAINEE) =====================

@training_bp.route("/training/submit", methods=["POST"])
def training_submit():
    if "training_employee_id" not in session:
        abort(401)

    ensure_schema()

    employee_id = session["training_employee_id"]

    if is_submitted(employee_id):
        return redirect(url_for("training.confirmation"))

    db = get_db()
    cur = db.cursor()
    cur.execute("""
        UPDATE employees
        SET submitted = 1,
            submitted_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (employee_id,))
    db.commit()
    db.close()

    return redirect(url_for("training.confirmation"))


@training_bp.route("/training/confirmation")
def confirmation():
    if "training_employee_id" not in session:
        abort(401)

    employee_id = session["training_employee_id"]

    if not is_submitted(employee_id):
        role = session.get("training_employee_role")
        if role == "YAS":
            return redirect(url_for("yas_booklet.yas_page", page=1))
        if role == "YSS":
            return redirect(url_for("yss_booklet.yss_page", page=1))
        return redirect(url_for("home.role_selection"))

    return render_template("training/confirmation.html")


# ===================== AUTOSAVE =====================

@training_bp.route("/training/autosave", methods=["POST"])
def training_autosave():
    data = request.get_json(force=True)

    page_code = data.get("page_code")
    field_name = data.get("field")
    field_value = data.get("value", "")

    if not page_code or not field_name:
        return {"status": "bad request"}, 400

    # ---- Determine who we are saving for ----
    # Trainees save for themselves
    # Mentors save for a target trainee (must pass target_employee_id)
    if is_mentor():
        target_employee_id = data.get("target_employee_id")
        if not target_employee_id:
            return {"status": "bad request", "error": "missing target_employee_id"}, 400
        employee_id = int(target_employee_id)
    else:
        if "training_employee_id" not in session:
            return {"status": "unauthorized"}, 401
        employee_id = int(session["training_employee_id"])

    # ---- Load target employee role + submitted flag ----
    ensure_schema()
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, role, submitted FROM employees WHERE id = ?", (employee_id,))
    emp = cur.fetchone()
    if not emp:
        db.close()
        return {"status": "not found"}, 404

    trainee_role = (emp["role"] or "").upper()
    submitted = int(emp["submitted"] or 0)

    # ---- Enforce locks ----
    # ✅ Trainee: locked on certain pages (even before submit)
    # ✅ Trainee: locked after submitted
    # ✅ Mentor: editable only on allowed pages (even if submitted)
    if not is_mentor():
        if trainee_page_locked(trainee_role, page_code):
            db.close()
            return {"status": "locked", "error": "page locked for trainee"}, 403

        if submitted == 1:
            db.close()
            return {"status": "locked"}, 403
    else:
        if not mentor_can_edit(trainee_role, page_code):
            db.close()
            return {"status": "locked", "error": "mentor not allowed on this page"}, 403

    # ---- Save ----
    cur.execute("""
        INSERT INTO training_progress
            (employee_id, page_code, field_name, field_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(employee_id, page_code, field_name)
        DO UPDATE SET
            field_value = excluded.field_value,
            updated_at = CURRENT_TIMESTAMP
    """, (employee_id, page_code, field_name, field_value))

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
    pages_dir, trainee_endpoint, template, booklet_title = _get_booklet_config(role)

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

        nav_ctx = _build_nav_context(
            admin_view=False,
            mentor_view=False,
            page=page,
            max_page=max_page,
            trainee_role=role,
            employee_id=None,
        )

        page_html = render_template(
            template,
            content_html=content_html,
            page=page,
            max_page=max_page,
            saved_data=saved_data.get(page_code, {}),
            booklet_title=booklet_title,
            booklet_endpoint=trainee_endpoint,
            admin_view=False,
            mentor_view=False,
            mentor_editable=False,
            target_employee_id=None,
            **nav_ctx,
        )
        pages_html.append(page_html)

    return render_template("yas_booklet/print_all.html", pages_html=pages_html)


# ============================================================
# ===================== ADMIN (existing) ======================
# ============================================================

@training_bp.route("/training/admin")
def admin_dashboard():
    if not is_admin():
        abort(403)

    ensure_schema()

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        SELECT
            e.id,
            e.name,
            e.role,
            e.submitted,
            datetime(e.submitted_at, '+8 hours') AS submitted_at,
            MAX(datetime(tp.updated_at, '+8 hours')) AS last_updated
        FROM employees e
        LEFT JOIN training_progress tp ON e.id = tp.employee_id
        GROUP BY e.id
        ORDER BY (last_updated IS NULL), last_updated DESC, e.name ASC
    """)

    trainees = cur.fetchall()
    db.close()

    return render_template("admin/index.html", trainees=trainees)


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
        SELECT
            page_code,
            COUNT(*) AS fields_filled,
            MAX(datetime(updated_at, '+8 hours')) AS last_updated
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

    pages_dir, trainee_endpoint, template, booklet_title = _get_booklet_config(trainee["role"])
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

    nav_ctx = _build_nav_context(
        admin_view=True,
        mentor_view=False,
        page=page,
        max_page=max_page,
        trainee_role=trainee["role"],
        employee_id=employee_id,
    )

    return render_template(
        template,
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data,
        booklet_title=booklet_title,
        booklet_endpoint=trainee_endpoint,
        admin_view=True,
        mentor_view=False,
        mentor_editable=False,
        target_employee_id=None,
        trainee_name=trainee["name"],
        trainee_role=trainee["role"],
        trainee_id=trainee["id"],
        **nav_ctx,
    )


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

    pages_dir, trainee_endpoint, template, booklet_title = _get_booklet_config(trainee["role"])
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

        nav_ctx = _build_nav_context(
            admin_view=True,
            mentor_view=False,
            page=page,
            max_page=max_page,
            trainee_role=trainee["role"],
            employee_id=employee_id,
        )

        page_html = render_template(
            template,
            content_html=content_html,
            page=page,
            max_page=max_page,
            saved_data=saved_data.get(page_code, {}),
            booklet_title=booklet_title,
            booklet_endpoint=trainee_endpoint,
            admin_view=True,
            mentor_view=False,
            mentor_editable=False,
            target_employee_id=None,
            trainee_name=trainee["name"],
            trainee_role=trainee["role"],
            trainee_id=trainee["id"],
            **nav_ctx,
        )
        pages_html.append(page_html)

    return render_template("yas_booklet/print_all.html", pages_html=pages_html)


# ============================================================
# ===================== MENTOR DASHBOARD ======================
# ============================================================

@training_bp.route("/training/mentor")
def mentor_dashboard():
    if not is_mentor():
        abort(403)

    ensure_schema()

    role_filter = (request.args.get("role") or "").strip().upper()
    if role_filter not in ("", "YAS", "YSS"):
        role_filter = ""

    q = (request.args.get("q") or "").strip()

    db = get_db()
    cur = db.cursor()

    where = []
    params = []

    # mentors view only trainees (YAS/YSS)
    where.append("e.role IN ('YAS','YSS')")

    if role_filter in ("YAS", "YSS"):
        where.append("e.role = ?")
        params.append(role_filter)

    if q:
        where.append("e.name LIKE ?")
        params.append(f"%{q}%")

    where_sql = " AND ".join(where) if where else "1=1"

    cur.execute(f"""
        SELECT
            e.id,
            e.name,
            e.role,
            e.submitted,
            datetime(e.submitted_at, '+8 hours') AS submitted_at,
            MAX(datetime(tp.updated_at, '+8 hours')) AS last_updated
        FROM employees e
        LEFT JOIN training_progress tp ON e.id = tp.employee_id
        WHERE {where_sql}
        GROUP BY e.id
        ORDER BY (last_updated IS NULL), last_updated DESC, e.name ASC
    """, params)

    trainees = cur.fetchall()
    db.close()

    return render_template("mentor/index.html", trainees=trainees, q=q, role_filter=role_filter)


@training_bp.route("/training/mentor/booklet/<int:employee_id>/<int:page>")
def mentor_view_page(employee_id, page):
    if not is_mentor():
        abort(403)

    ensure_schema()

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name, role, submitted FROM employees WHERE id = ?", (employee_id,))
    trainee = cur.fetchone()
    if not trainee:
        db.close()
        abort(404)

    pages_dir, trainee_endpoint, template, booklet_title = _get_booklet_config(trainee["role"])
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

    editable = mentor_can_edit(trainee["role"], page_code)

    nav_ctx = _build_nav_context(
        admin_view=False,
        mentor_view=True,
        page=page,
        max_page=max_page,
        trainee_role=trainee["role"],
        employee_id=employee_id,
    )

    return render_template(
        template,
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data,
        booklet_title=booklet_title,
        booklet_endpoint=trainee_endpoint,
        admin_view=False,
        mentor_view=True,
        mentor_editable=editable,
        target_employee_id=trainee["id"],
        trainee_name=trainee["name"],
        trainee_role=trainee["role"],
        trainee_id=trainee["id"],
        trainee_submitted=int(trainee["submitted"] or 0),
        **nav_ctx,
    )
