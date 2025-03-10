# tests/test_main.py

import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(ROOT_DIR)

import app.main as main
from app.database import get_db
from fastapi import HTTPException



class DummySession:
    async def commit(self):
        pass

    def add(self, obj):
        pass


# Override get_db dependency to yield our dummy session.
async def override_get_db():
    yield DummySession()


main.app.dependency_overrides[get_db] = override_get_db


# Dummy implementations to override external calls.
def dummy_hybrid_search(query: str, top_k: int = 5):
    return [{"document_id": "doc1", "document_name": "Test Document"}]


def dummy_generate_roadmap_batch(document_id: str, user_request: str):
    return "Dummy Roadmap Text"


def dummy_generate_roadmap_batch_error(document_id: str, user_request: str):
    raise HTTPException(status_code=404, detail="Document not found")


@pytest.fixture(autouse=True)
def patch_functions(monkeypatch):
    # Patch the functions as they were imported in main.py.
    monkeypatch.setattr(main, "hybrid_search", dummy_hybrid_search)
    monkeypatch.setattr(main, "generate_roadmap_batch", dummy_generate_roadmap_batch)


client = TestClient(main.app)


def test_search_endpoint():
    response = client.get("/search", params={"query": "test", "top_k": 1})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["query"] == "test"
    assert json_data["results"] == [{"document_id": "doc1", "document_name": "Test Document"}]


def test_search_endpoint_missing_query():
    # Omitting the required query parameter should trigger validation.
    response = client.get("/search")
    assert response.status_code == 422


def test_generate_roadmap_endpoint():
    payload = {"document_id": "doc1", "user_request": "Test request"}
    response = client.post("/generate-roadmap/", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["roadmap"] == "Dummy Roadmap Text"


def test_generate_roadmap_not_found(monkeypatch):
    monkeypatch.setattr(main, "generate_roadmap_batch", dummy_generate_roadmap_batch_error)
    payload = {"document_id": "non_existent_doc", "user_request": "Test request"}
    response = client.post("/generate-roadmap/", json=payload)
    assert response.status_code == 404
    json_data = response.json()
    assert json_data["detail"] == "Document not found"
