from fastapi.testclient import TestClient
from runway_radar.api.main import app

client = TestClient(app)

# test health check endpoint

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "oki!"}

# test seasons endpoint

def test_get_seasons():
    response = client.get("/seasons")
    assert response.status_code == 200
    assert len(response.json()) == 16
    assert response.json()[0] == {"season_index": 1, "season": "Fall", "year": 2015}
    assert response.json()[-1]["season_index"] == 16

# test trends endpoint

def test_trends_no_filters():
    response = client.get("/trends")
    assert response.status_code == 200
    assert len(response.json()) == 108

def test_trends_with_filter():
    response = client.get("/trends?axis=pattern&line=mens")
    assert response.status_code == 200
    assert all(row["axis"] == "pattern" and row["line"] == "mens" for row in response.json())

def test_trends_invalid_axis_filter():
    response = client.get("/trends?axis=kids")
    assert response.status_code == 422

# test prevalence endpoint
def test_prevalence_no_filters():
    response = client.get("/prevalence")
    assert response.status_code == 200
    assert len(response.json()) == 1728

def test_prevalence_with_filter():
    response = client.get("/prevalence?axis=pattern&line=mens&label=floral_print")
    assert response.status_code == 200
    assert all(row["axis"] == "pattern" and row["line"] == "mens" and row["label"] == "floral_print" for row in response.json())
    assert len(response.json()) == 16

def test_prevalence_is_ordered_by_season():
    response = client.get("/prevalence?axis=pattern&line=mens&label=floral_print")
    indexes = [row["season_index"] for row in response.json()]
    assert indexes == sorted(indexes)