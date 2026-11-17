import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "runway.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # this allows us to access the columns by name instead of by index
    try:
        yield conn
    finally:
        conn.close()

    # what try-finally does is that it ensures that the connection is closed even if an exception occurs
    # yield is used to return the connection object to the caller
    # and then the finally block is executed after the caller is done with the connection
