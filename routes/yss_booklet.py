# routes/yss_booklet.py

import sqlite3
from pathlib import Path
from flask import (
    Blueprint, render_template, abort,
    redirect, url_for, session
)

yss_booklet_bp = Blueprint("yss_booklet", __name__)

# ===================== PATHS =====================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "training.db"
PAGES_DIR = BASE_DIR / "OJT" / "yss_booklet" / "pages"

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

@yss_booklet_bp.route("/yss")
def yss_index():
    return redirect(url_for("yss_booklet.yss_page", page=1))


@yss_booklet_bp.route("/yss/<int:page>")
def yss_page(page: int):

    # 🔒 TRAINING SESSION CHECK
    if not require_training_login():
        return redirect(url_for("home.role_selection"))

    max_page = _max_page()
    if max_page == 0:
        abort(404, description="No YSS booklet pages found.")

    if page < 1 or page > max_page:
        abort(404)

    page_path = PAGES_DIR / f"{page:03d}.html"
    if not page_path.exists():
        abort(404)

    content_html = page_path.read_text(encoding="utf-8")

    # 🔄 Load saved field values
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
        "yas_booklet/page.html",
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data,
        booklet_title="YSS Booklet",
        booklet_endpoint="yss_booklet.yss_page"
    )
