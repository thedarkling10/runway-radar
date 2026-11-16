from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Literal

import sqlite3

from runway_radar.api.db import get_db

app = FastAPI(title="Runway Radar API")

Axis = Literal["pattern", "color", "embellishment", "silhouette", "neckline_collar", "fabric_texture"]
Line = Literal["mens", "womens"]

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

class Trend(BaseModel):
    line: str
    axis: str
    label: str
    mean_prevalence: float
    slope: float
    r_squared: float | None
    p_value: float | None
    direction: str
    projected_next_season: float


@app.get("/trends", response_model=list[Trend])
def get_trends(axis:Axis | None = None, 
               line:Line | None = None, conn: sqlite3.Connection=Depends(get_db)):
    query = "SELECT * FROM trend_summary"
    conditions = []
    params = []

    if axis:
        conditions.append("axis = ?")
        params.append(axis)
    if line:
        conditions.append("line = ?")
        params.append(line)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]    

class Prevalence(BaseModel):
    season_index: int
    season: str
    year: int
    line: str
    axis: str
    label: str
    count: int
    n_images: int
    prevalence: float
    
@app.get("/prevalence", response_model=list[Prevalence])
def get_prevalence(season_index: int | None = None,
                   line:Line | None = None,
                   axis:Axis | None = None,
                   label: str | None = None,
                   conn: sqlite3.Connection=Depends(get_db)):
    query = "SELECT * FROM seasonal_prevalence"
    conditions = []
    params = []

    if season_index is not None:
        conditions.append("season_index = ?")
        params.append(season_index)
    if line:
        conditions.append("line = ?")
        params.append(line)
    if axis:
        conditions.append("axis = ?")
        params.append(axis)
    if label:
        conditions.append("label = ?")
        params.append(label)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY season_index"

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

