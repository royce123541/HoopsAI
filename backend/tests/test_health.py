from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from hoopsai.api.main import create_app
from hoopsai.api.routers import health


async def _ok(*_: object) -> None:
    return None


async def _fail(*_: object) -> None:
    raise ConnectionError("down")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        yield c


def test_health_ok_when_dependencies_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_check_database", _ok)
    monkeypatch.setattr(health, "_check_redis", _ok)

    resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "redis": "ok"}


def test_health_degraded_but_200_when_dependency_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_check_database", _fail)
    monkeypatch.setattr(health, "_check_redis", _ok)

    resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"database": "error: ConnectionError", "redis": "ok"}
