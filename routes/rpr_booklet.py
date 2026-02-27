# routes/<new_booklet>.py
# (Copy from routes/yss_booklet.py)

import sqlite3
from pathlib import Path
from flask import (
    Blueprint, render_template, abort,
    redirect, url_for, session
)

# ✅ CHANGE 1: Blueprint name (both variable + blueprint string)
# Example: yxx_booklet_bp = Blueprint("yxx_booklet", __name__)
rpr_booklet_bp = Blueprint("rpr_booklet", __name__)  # <-- CHANGE

# ===================== PATHS =====================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "training.db"

# ✅ CHANGE 2: PAGES_DIR folder path (must match your new booklet folder)
# Example: BASE_DIR / "OJT" / "yxx_booklet" / "pages"
PAGES_DIR = BASE_DIR / "OJT" / "rpr_booklet" / "pages"  # <-- CHANGE

# ===================== HELPERS =====================

def require_training_login():
    return "training_employee_id" in session


def _max_page() -> int:
    if not PAGES_DIR.exists():
        return 0

    pages = sorted(PAGES_DIR.glob("*.html"))
    nums = []
    for p in pages:
        try:
            nums.append(int(p.stem))
        except ValueError:
            pass

    return max(nums) if nums else 0


# ===================== ROUTES =====================

# ✅ CHANGE 3: URL prefix + function name + endpoint reference inside url_for()
# Example route: "/yxx"
# Example function: yxx_index()
# Example redirect target: "yxx_booklet.yxx_page"
@rpr_booklet_bp.route("/rpr")  # <-- CHANGE
def rpr_index():               # <-- CHANGE
    return redirect(url_for("rpr_booklet.rpr_page", page=1))  # <-- CHANGE


# ✅ CHANGE 4: URL prefix + function name
# Example route: "/yxx/<int:page>"
# Example function: yxx_page(page)
@rpr_booklet_bp.route("/rpr/<int:page>")  # <-- CHANGE
def rpr_page(page: int):                  # <-- CHANGE

    # 🔒 TRAINING SESSION CHECK (usually keep the same)
    if not require_training_login():
        return redirect(url_for("home.role_selection"))

    max_page = _max_page()
    if max_page == 0:
        abort(404, description="No RPM booklet pages found.")  # (optional) CHANGE TEXT only

    if page < 1 or page > max_page:
        abort(404)

    page_path = PAGES_DIR / f"{page:03d}.html"
    if not page_path.exists():
        abort(404)

    content_html = page_path.read_text(encoding="utf-8")

    # 🔄 Load saved field values (usually keep the same)
    saved_data = {}
    if "training_employee_id" in session:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
            SELECT field_name, field_value
            FROM training_progress
            WHERE employee_id = ? AND page_code = ?
        """, (
            session["training_employee_id"],
            f"{page:03d}"
        ))

        saved_data = dict(cur.fetchall())
        conn.close()

    return render_template(
        "yas_booklet/page.html",  # (usually keep the same shared template)
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data,

        # ✅ CHANGE 5: Title shown in browser/tab + header (booklet_title)
        booklet_title="RP Booklet(Replanner)",  # <-- CHANGE to "NEW Booklet"

        # ✅ CHANGE 6: Endpoint used by page.html for Next/Prev/Jump
        # Must match: "<blueprint_name>.<page_route_function>"
        # Example: "yxx_booklet.yxx_page"
        booklet_endpoint="rpr_booklet.rpr_page"  # <-- CHANGE
    )
