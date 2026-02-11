import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "training.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def find_employee_id(conn, name: str, role: str) -> int | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM employees WHERE name = ? AND role = ?",
        (name, role),
    )
    row = cur.fetchone()
    return row["id"] if row else None

def preview_control_chars(conn, employee_id: int, page_code: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT field_name, field_value
        FROM training_progress
        WHERE employee_id = ? AND page_code = ?
        ORDER BY field_name
        """,
        (employee_id, page_code),
    )
    rows = cur.fetchall()

    bad = []
    for r in rows:
        v = r["field_value"] or ""
        has_lf = "\n" in v
        has_cr = "\r" in v
        has_tab = "\t" in v
        if has_lf or has_cr or has_tab:
            preview = v[:120].replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
            bad.append((r["field_name"], has_lf, has_cr, has_tab, preview))

    print(f"Employee {employee_id} page {page_code}: {len(rows)} rows")
    if not bad:
        print("No LF/CR/TAB found ✅")
        return

    print("\nFOUND control chars (these used to crash JSON.parse):")
    for field_name, lf, cr, tab, pv in bad:
        print(f" - {field_name}: LF={lf} CR={cr} TAB={tab} preview={pv}")

def sanitize_page(conn, employee_id: int, page_code: str):
    """
    Replace CR/LF/TAB with a single space, keep content.
    """
    cur = conn.cursor()

    # SQLite replace() is simple and safe
    cur.execute(
        """
        UPDATE training_progress
        SET field_value =
            replace(
              replace(
                replace(field_value, char(13), ' '),
              char(10), ' '),
            char(9), ' ')
        WHERE employee_id = ? AND page_code = ?
        """,
        (employee_id, page_code),
    )

    conn.commit()
    print(f"Sanitized page {page_code} for employee {employee_id}. Rows updated: {cur.rowcount}")

def wipe_page(conn, employee_id: int, page_code: str):
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM training_progress WHERE employee_id = ? AND page_code = ?",
        (employee_id, page_code),
    )
    conn.commit()
    print(f"Wiped page {page_code} for employee {employee_id}. Rows deleted: {cur.rowcount}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Employee login name (exact)")
    parser.add_argument("--role", default="YSS", help="Role (default: YSS)")
    parser.add_argument("--employee-id", type=int, help="Employee ID (if you already know it)")
    parser.add_argument("--page", default="011", help="Page code like 011 (default: 011)")
    parser.add_argument("--action", choices=["preview", "sanitize", "wipe"], default="preview")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    conn = get_conn()
    try:
        emp_id = args.employee_id
        if emp_id is None:
            if not args.name:
                raise SystemExit("Provide either --employee-id or --name")
            emp_id = find_employee_id(conn, args.name, args.role)
            if emp_id is None:
                raise SystemExit(f"No employee found for name={args.name!r}, role={args.role!r}")

        if args.action == "preview":
            preview_control_chars(conn, emp_id, args.page)
        elif args.action == "sanitize":
            sanitize_page(conn, emp_id, args.page)
            preview_control_chars(conn, emp_id, args.page)
        elif args.action == "wipe":
            wipe_page(conn, emp_id, args.page)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
