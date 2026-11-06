import secrets

import pytest
from fastapi.testclient import TestClient

from hydrobrain import HydroBrain
from hydrobrain.sessions import SessionStore


INPUTS = dict(lat=49.23, lon=28.47, crop="winter_wheat", sowing_date="2026-10-01",
              irrigation_method="drip", soil_type_ua="chornozem_typical")
OPERATIONS = [
    ("GET", "info", None),
    ("PATCH", "", {"area_m2": 500}),
    ("POST", "sensor", {"theta": 0.27}),
    ("POST", "irrigation", {"amount_mm": 5}),
    ("POST", "fertilizer", {"dose_kg_ha": 2}),
    ("POST", "calibrate_soil", None),
    ("POST", "finetune_local", None),
    ("POST", "recommendation", None),
    ("GET", "recommendation", None),
]


@pytest.fixture
def api(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "brain", HydroBrain(str(tmp_path), offline=True))
    monkeypatch.setattr(server, "OFFLINE", True)
    with TestClient(server.app) as client:
        yield server, client


def visitor(client):
    response = client.post("/sessions")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    token = response.json()["token"]
    return token, {"Authorization": "Bearer " + token}


def new_field(client, headers):
    result = client.post("/fields", headers=headers, json={"inputs": INPUTS, "name": "Private field"})
    assert result.status_code == 200, result.text
    return result.json()["field_id"]


def test_two_visitors_have_isolated_lists(api):
    server, client = api
    _, a = visitor(client)
    _, b = visitor(client)
    fa, fb = new_field(client, a), new_field(client, b)
    for headers, own, other in [(a, fa, fb), (b, fb, fa)]:
        assert client.get("/fields", headers=headers).json() == [own]
        response = client.get("/fields_detailed", headers=headers)
        assert [item["field_id"] for item in response.json()] == [own]
        assert "no-store" in response.headers["Cache-Control"]
        assert "Authorization" in response.headers["Vary"]
        assert client.get(f"/fields/{own}/info", headers=headers).status_code == 200
        assert client.get(f"/fields/{other}/info", headers=headers).status_code == 404


@pytest.mark.parametrize("method,suffix,body", OPERATIONS)
def test_foreign_field_is_hidden_and_unchanged(api, method, suffix, body):
    server, client = api
    _, a = visitor(client)
    _, b = visitor(client)
    fid = new_field(client, a)
    before = server.brain.load_field(fid)
    path = f"/fields/{fid}" + ("/" + suffix if suffix else "")
    response = client.request(method, path, headers=b, json=body)
    missing = client.request(method, path.replace(fid, "missing"), headers=b, json=body)
    assert response.status_code == missing.status_code == 404
    assert response.json() == missing.json() == {"detail": "Поле не знайдено"}
    assert server.brain.load_field(fid) == before


@pytest.mark.parametrize("method,suffix,body", OPERATIONS)
def test_each_field_operation_requires_session(api, method, suffix, body):
    server, client = api
    _, a = visitor(client)
    fid = new_field(client, a)
    path = f"/fields/{fid}" + ("/" + suffix if suffix else "")
    assert client.request(method, path, json=body).status_code == 401


def test_no_anonymous_list_or_creation_and_no_forged_token(api):
    _, client = api
    for headers in [{}, {"Authorization": "Basic fake"}, {"Authorization": "Bearer ../outside"},
                    {"Authorization": "Bearer " + secrets.token_urlsafe(32)}]:
        assert client.get("/fields", headers=headers).status_code == 401
        assert client.get("/fields_detailed", headers=headers).status_code == 401
        assert client.get("/sessions/current", headers=headers).status_code == 401
        assert client.post("/fields", headers=headers, json={"inputs": INPUTS}).status_code == 401


def test_ownership_and_token_survive_new_store_and_client(api):
    server, client = api
    token, a = visitor(client)
    fid = new_field(client, a)
    store = SessionStore(server.brain.data_dir)
    digest = store.authenticate(token)
    assert digest and digest != token and store.owns(fid, digest)
    assert token.encode() not in store.path.read_bytes()
    server._session_store.cache_clear()
    with TestClient(server.app) as restored:
        assert restored.get("/fields", headers=a).json() == [fid]
        assert restored.get("/sessions/current", headers=a).json() == {"ok": True}


def test_legacy_shared_fields_are_preserved_but_not_claimable(api):
    server, client = api
    legacy = server.brain.create_field(INPUTS)["field_id"]
    _, a = visitor(client)
    _, b = visitor(client)
    for headers in (a, b):
        assert client.get("/fields", headers=headers).json() == []
        assert client.get("/fields_detailed", headers=headers).json() == []
        assert client.get(f"/fields/{legacy}/info", headers=headers).status_code == 404
    assert server.brain.load_field(legacy)["field_id"] == legacy


def test_owner_can_save_events_and_edit(api):
    server, client = api
    _, a = visitor(client)
    fid = new_field(client, a)
    assert client.patch(f"/fields/{fid}", headers=a, json={"area_m2": 600}).status_code == 200
    assert client.post(f"/fields/{fid}/sensor", headers=a, json={"theta": 27}).status_code == 200
    assert client.post(f"/fields/{fid}/irrigation", headers=a, json={"amount_mm": 5}).status_code == 200
    assert client.post(f"/fields/{fid}/fertilizer", headers=a, json={"dose_kg_ha": 2}).status_code == 200
    info = client.get(f"/fields/{fid}/info", headers=a).json()
    assert info["summary"]["area_m2"] == 600
    assert all(len(info["events"][kind]) == 1 for kind in ("sensor", "irrigation", "fertilizer"))


def test_public_resources_and_authorization_preflight(api):
    _, client = api
    for path in ("/", "/session.js", "/catalog", "/schema/beginner"):
        assert client.get(path).status_code == 200
    response = client.options("/fields", headers={"Origin": "https://kokojas.github.io",
        "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization,content-type"})
    assert response.status_code == 200
    assert "authorization" in response.headers["Access-Control-Allow-Headers"].lower()
