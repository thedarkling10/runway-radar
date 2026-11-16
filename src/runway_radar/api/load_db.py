# defining the repo root
from pathlib import Path
import pandas as pd
import sqlite3

def main():
    repo_root = Path(__file__).resolve().parents[3]
    db_path = repo_root / "data" / "runway.db"

    #print(f"repo root: {repo_root}")
    #print(f"db path: {db_path}")


    # turning csvs to dataframes --> they will become tables in db
    df_prevalence = pd.read_csv(repo_root / "data" / "processed" / "seasonal_prevalence.csv")
    df_trend = pd.read_csv(repo_root / "data" / "processed" / "trend_summary.csv")
    df_colors = pd.read_csv(repo_root / "data" / "processed" / "colors.csv")
    df_looks = pd.read_csv(repo_root / "data" / "metadata" / "looks.csv")

    conn = sqlite3.connect(db_path)

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


