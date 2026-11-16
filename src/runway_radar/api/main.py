from fastapi import FastAPI

app = FastAPI(title="Runway Radar API")

@app.get("/health")
def health_check():
    return {"status": "oki!"}