from fastapi.testclient import TestClient
from runway_radar.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "oki!"}

def test_get_seasons():
    response = client.get("/seasons")
    assert response.status_code == 200
    assert len(response.json()) == 16
    assert response.json()[0] == {"season_index": 1, "season": "Fall", "year": 2015}
    assert response.json()[-1]["season_index"] == 16