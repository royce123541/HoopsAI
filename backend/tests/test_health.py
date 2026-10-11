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


def test_health_ok_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serverless hosts run without Redis; live charts poll instead."""
    from hoopsai.config import get_settings

    monkeypatch.setattr(get_settings(), "redis_url", "")
    monkeypatch.setattr(health, "_check_database", _ok)

    with TestClient(create_app()) as c:
        body = c.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "redis": "disabled"}


def test_redis_can_be_switched_off() -> None:
    from hoopsai.config import Settings

    assert Settings(redis_url="off").redis_url == ""


def test_neon_style_urls_use_psycopg() -> None:
    from hoopsai.config import Settings

    url = "postgresql://u:p@ep-x.us-east-1.aws.neon.tech/neondb?sslmode=require"
    assert Settings(database_url=url).database_url == (
        "postgresql+psycopg://u:p@ep-x.us-east-1.aws.neon.tech/neondb?sslmode=require"
    )
