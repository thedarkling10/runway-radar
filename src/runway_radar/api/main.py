from fastapi import Depends, FastAPI
from pydantic import BaseModel

import sqlite3
from runway_radar.api.db import get_db

app = FastAPI(title="Runway Radar API")

@app.get("/health")
def health_check():
    return {"status": "oki!"}

class Season(BaseModel):
    season_index: int
    season: str
    year: int

@app.get("/seasons", response_model=list[Season])
def get_seasons(conn: sqlite3.Connection=Depends(get_db)):
    rows = conn.execute("SELECT DISTINCT season_index, season, year FROM looks ORDER BY season_index").fetchall()
    return [dict(row) for row in rows]
    
