import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "training.db"

name = "Tay Johan"   # <-- change
role = "YSS"          # <-- change if needed

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 1) get employee
cur.execute("""
SELECT id, name, role, submitted
FROM employees
WHERE name = ? AND role = ?
""", (name, role))
emp = cur.fetchone()

if not emp:
    print("No employee found for", name, role)
    raise SystemExit

emp_id = emp["id"]
print("Employee:", dict(emp))

# 2) show page 011 rows
cur.execute("""
SELECT page_code, field_name, field_value
FROM training_progress
WHERE employee_id = ? AND page_code = '011'
ORDER BY field_name
""", (emp_id,))
rows = cur.fetchall()
print("\nPage 011 rows:", len(rows))

# 3) detect control chars
bad = []
for r in rows:
    v = r["field_value"] or ""
    has_lf = "\n" in v
    has_cr = "\r" in v
    has_tab = "\t" in v
    if has_lf or has_cr or has_tab:
        bad.append((r["field_name"], has_lf, has_cr, has_tab, v[:80].replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")))

if bad:
    print("\nFOUND control characters (this causes JSON.parse crash):")
    for item in bad:
        print(" -", item[0], "LF:", item[1], "CR:", item[2], "TAB:", item[3], "preview:", item[4])
else:
    print("\nNo LF/CR/TAB found in page 011 values.")

conn.close()
