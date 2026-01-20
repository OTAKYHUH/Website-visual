import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "training.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT id, name, role, created_at FROM employees")

rows = cur.fetchall()

print("Employees table:")
for r in rows:
    print(r)

conn.close()
