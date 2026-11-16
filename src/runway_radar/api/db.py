import sqlite3
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
db_path = repo_root / "data" / "runway.db"

def get_db():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # this allows us to access the columns by name instead of by index
    try:
        yield conn
    finally:
        conn.close()

    # what try-finally does is that it ensures that the connection is closed even if an exception occurs
    # yield is used to return the connection object to the caller
    # and then the finally block is executed after the caller is done with the connection

    # for example, if a caller (e.g., a FastAPI endpoint) uses this function like this:
    # with get_db() as conn:
    #     # do something with conn
    # then the connection will be closed automatically after the with block is exited, even if an exception occurs