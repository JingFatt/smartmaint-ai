import pytest
from fastapi.testclient import TestClient

from backend.main import app

NORMAL = dict(TP2=1.0, TP3=9.0, H1=8.0, DV_pressure=0.0, Reservoirs=9.0, Oil_temperature=62.0,
              Motor_current=2.0, COMP=1.0, DV_eletric=0.0, Towers=1.0, MPG=1.0, LPS=0.0,
              Pressure_switch=1.0, Oil_level=1.0, Caudal_impulses=1.0)
FAULT = dict(TP2=8.0, TP3=8.0, H1=0.0, DV_pressure=2.0, Reservoirs=8.0, Oil_temperature=75.0,
             Motor_current=5.5, COMP=0.0, DV_eletric=1.0, Towers=1.0, MPG=0.0, LPS=0.0,
             Pressure_switch=1.0, Oil_level=1.0, Caudal_impulses=1.0)


def payload(row, n):
    return {"equipment_id": "test-001", "readings": [row] * n}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    body = client.get("/model-info").json()
    assert len(body["features"]) == 15
    assert body["alias"] == "champion"
    assert len(body["known_limitations"]) > 0


def test_normal_window_no_rule_alarm(client):
    r = client.post("/predict", json=payload(NORMAL, 180))
    assert r.status_code == 200
    body = r.json()
    assert body["window_complete"] is True
    assert body["load_fraction"] == 0.0
    assert body["rule_alarm"] is False


def test_fault_window_rule_alarm(client):
    r = client.post("/predict", json=payload(FAULT, 180))
    assert r.status_code == 200
    body = r.json()
    assert body["load_fraction"] == 1.0
    assert body["rule_alarm"] is True


def test_short_window_has_null_alarms(client):
    body = client.post("/predict", json=payload(NORMAL, 10)).json()
    assert body["window_complete"] is False
    assert body["if_alarm"] is None
    assert body["rule_alarm"] is None


def test_missing_field_rejected(client):
    row = {k: v for k, v in NORMAL.items() if k != "TP2"}
    assert client.post("/predict", json=payload(row, 5)).status_code == 422


def test_empty_readings_rejected(client):
    assert client.post("/predict", json={"equipment_id": "x", "readings": []}).status_code == 422


def test_extra_field_rejected(client):
    row = {**NORMAL, "unknown_sensor": 1.0}
    assert client.post("/predict", json=payload(row, 5)).status_code == 422