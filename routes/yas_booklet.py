from pathlib import Path
import sqlite3
from flask import Blueprint, render_template, abort, redirect, url_for, session

yas_booklet_bp = Blueprint("yas_booklet", __name__)

# Adjust if your project structure differs
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "training.db"
PAGES_DIR = Path(__file__).resolve().parents[1] / "OJT" / "yas_booklet" / "pages"

def require_training_login():
    if "training_employee_id" not in session:
        return False
    return True

def _max_page() -> int:
    if not PAGES_DIR.exists():
        return 0
    # expects files like 001.html, 002.html...
    pages = sorted(PAGES_DIR.glob("*.html"))
    nums = []
    for p in pages:
        try:
            nums.append(int(p.stem))
        except ValueError:
            pass
    return max(nums) if nums else 0


@yas_booklet_bp.route("/yas")
def yas_index():
    return redirect(url_for("yas_booklet.yas_page", page=1))


@yas_booklet_bp.route("/yas/<int:page>")
def yas_page(page: int):

    # 🔒 TRAINING SESSION CHECK
    if not require_training_login():
        return redirect(url_for("home.role_selection"))

    max_page = _max_page()
    if max_page == 0:
        abort(404, description="No booklet pages found.")
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
        """, (session["training_employee_id"], f"{page:03d}"))

        saved_data = dict(cur.fetchall())
        conn.close()

    return render_template(
        "yas_booklet/page.html",
        content_html=content_html,
        page=page,
        max_page=max_page,
        saved_data=saved_data
    )
