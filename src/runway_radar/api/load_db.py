import pandas as pd
import sqlite3
from runway_radar.api.db import DB_PATH
from runway_radar.api.db import REPO_ROOT

def main():

    #print(f"repo root: {REPO_ROOT}")
    #print(f"db path: {DB_PATH}")


    # turning csvs to dataframes --> they will become tables in db
    df_prevalence = pd.read_csv(REPO_ROOT / "data" / "processed" / "seasonal_prevalence.csv")
    df_trend = pd.read_csv(REPO_ROOT / "data" / "processed" / "trend_summary.csv")
    df_colors = pd.read_csv(REPO_ROOT / "data" / "processed" / "colors.csv")
    df_looks = pd.read_csv(REPO_ROOT / "data" / "metadata" / "looks.csv")

    conn = sqlite3.connect(DB_PATH)

    df_prevalence.to_sql("seasonal_prevalence", conn, if_exists="replace", index=False)
    df_trend.to_sql("trend_summary", conn, if_exists="replace", index=False)
    df_colors.to_sql("colors", conn, if_exists="replace", index=False)
    df_looks.to_sql("looks", conn, if_exists="replace", index=False)

    for table_name in ["seasonal_prevalence", "trend_summary", "colors", "looks"]:
        rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Table '{table_name}' has {rows} rows")

    conn.close()

if __name__ == "__main__":
    main()


