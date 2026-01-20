import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "training.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, role)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS training_progress (
    employee_id INTEGER NOT NULL,
    page_code TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (employee_id, page_code, field_name),
    FOREIGN KEY (employee_id)
      REFERENCES employees(id)
      ON DELETE CASCADE
);
""")

conn.commit()
conn.close()

print("✅ training.db initialized at", DB_PATH)