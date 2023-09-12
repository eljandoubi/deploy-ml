import json
from fastapi.testclient import TestClient

from main import app, features

client = TestClient(app)


def test_get():
    r=client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to Census Bureau Classifier API"

def test_post_0():
    data = json.dumps(features.Config.json_schema_extra[0])
    r = client.post("/invocations", data=data)
    assert r.status_code == 200
    assert r.json()=="<=50K"
    
def test_post_1():
    data = json.dumps(features.Config.json_schema_extra[1])
    r = client.post("/invocations", data=data)
    assert r.status_code == 200
    assert r.json()==">50K"


