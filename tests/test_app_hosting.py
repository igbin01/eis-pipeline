"""Hosting behaviour of the web app (app/server.py).

Configuration is read from ``EISPIPE_*`` environment variables at import time,
so every test loads a fresh copy of the server module with its own environment.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import itertools
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from flask.testing import FlaskClient

SERVER_PATH = Path(__file__).resolve().parents[1] / "app" / "server.py"
API_HEADERS = {"X-Requested-With": "eispipe"}
_module_names = itertools.count()


class _LocalClient(FlaskClient):
    """Defaults to the local Host; an explicit Host header still overrides it."""

    def open(self, *args, **kwargs):
        kwargs.setdefault("base_url", "http://localhost:5050")
        return super().open(*args, **kwargs)


@pytest.fixture
def load_server(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    loaded = []

    def _load(**env):
        for key in [k for k in os.environ if k.startswith("EISPIPE_")]:
            monkeypatch.delenv(key)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        name = f"eis_app_server_hosting_{next(_module_names)}"
        spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module  # Flask derives root_path from sys.modules
        loaded.append(name)
        spec.loader.exec_module(module)
        module.app.config["TESTING"] = True
        module.app.test_client_class = _LocalClient
        return module

    yield _load
    for name in loaded:
        sys.modules.pop(name, None)


def _basic(password: str, user: str = "anyone") -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _demo_job(client, headers=None) -> str:
    r = client.post(
        "/api/upload", data={"use_demo_data": "1"}, headers={**API_HEADERS, **(headers or {})}
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["job_id"]


def _fast_pipeline(monkeypatch, server, fit=None):
    """Replace the expensive stages so a run takes milliseconds (plot then fails)."""
    monkeypatch.setattr(server, "validate_series", lambda series: pd.DataFrame())
    monkeypatch.setattr(server, "fit_series", fit or (lambda *a, **k: []))
    monkeypatch.setattr(server, "metrics_table", lambda fits: pd.DataFrame())


def _events(stream: str) -> list[dict]:
    return [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------------------
# local defaults
# ---------------------------------------------------------------------------


def test_local_defaults_unchanged(load_server):
    server = load_server()

    assert server.ALLOWED_HOSTS == frozenset({"127.0.0.1:5050", "localhost:5050"})
    assert server.PASSWORD is None
    assert server.TRUST_PROXY is False
    assert server.MAX_JOBS == 2
    assert server.UPLOADS_PER_MINUTE == 10
    assert server.app.config["MAX_CONTENT_LENGTH"] == 25 * 1024 * 1024
    assert server.MAX_SPECTRUM_FILES == 64
    assert server.JOB_TIMEOUT_S == 300
    assert server.HEARTBEAT_S == 15
    client = server.app.test_client()
    assert client.get("/").status_code == 200  # no password needed
    # the dev server never listens publicly
    assert 'app.run(host="127.0.0.1", port=5050)' in SERVER_PATH.read_text()


# ---------------------------------------------------------------------------
# password gate and /healthz
# ---------------------------------------------------------------------------


def test_password_gate_off_by_default(load_server):
    server = load_server()
    client = server.app.test_client()

    assert client.get("/").status_code == 200
    job_id = _demo_job(client)
    assert client.get(f"/api/run/{job_id}", buffered=True).status_code == 200


def test_password_gate_on(load_server, monkeypatch):
    server = load_server(EISPIPE_PASSWORD="s3cret pass")
    _fast_pipeline(monkeypatch, server)
    client = server.app.test_client()

    r = client.get("/")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Basic")
    assert r.headers["X-Frame-Options"] == "DENY"  # security headers on 401 too
    assert client.get("/", headers=_basic("wrong")).status_code == 401
    assert client.get("/", headers=_basic("s3cret pass", user="whoever")).status_code == 200

    assert client.post("/api/upload", data={"use_demo_data": "1"}, headers=API_HEADERS).status_code == 401
    job_id = _demo_job(client, headers=_basic("s3cret pass"))
    assert client.get(f"/api/run/{job_id}").status_code == 401
    assert client.get(f"/api/run/{job_id}", headers=_basic("s3cret pass")).status_code == 200


@pytest.mark.parametrize("env", [{}, {"EISPIPE_PASSWORD": "s3cret"}])
def test_healthz_is_open(load_server, env):
    server = load_server(**env)
    client = server.app.test_client()

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_data(as_text=True) == "ok"
    # not under /api/, so not subject to the Host allow-list either
    assert client.get("/healthz", headers={"Host": "probe.internal:8080"}).status_code == 200


# ---------------------------------------------------------------------------
# Host / Origin allow-list from the environment
# ---------------------------------------------------------------------------


def test_allowed_hosts_from_env_accepts_https_origin(load_server):
    server = load_server(EISPIPE_ALLOWED_HOSTS="eis.example.org, localhost:5050")
    client = server.app.test_client()
    demo = {"use_demo_data": "1"}

    assert server.ALLOWED_HOSTS == frozenset({"eis.example.org", "localhost:5050"})
    host = {"Host": "eis.example.org", **API_HEADERS}
    assert client.post("/api/upload", data=demo,
                       headers={**host, "Origin": "https://eis.example.org"}).status_code == 200
    assert client.post("/api/upload", data=demo,
                       headers={**host, "Origin": "http://eis.example.org"}).status_code == 200
    assert client.post("/api/upload", data=demo,
                       headers={**host, "Origin": "https://evil.example"}).status_code == 403
    # the default local hosts are replaced, not extended
    assert client.post("/api/upload", data=demo,
                       headers={"Host": "127.0.0.1:5050", **API_HEADERS}).status_code == 403
    assert client.get("/api/run/x", headers={"Host": "evil.example"}).status_code == 403


# ---------------------------------------------------------------------------
# abuse limits
# ---------------------------------------------------------------------------


def test_job_cap_returns_503_while_jobs_run(load_server, monkeypatch):
    server = load_server(EISPIPE_MAX_JOBS="1")
    release = threading.Event()

    def blocking_fit(*args, **kwargs):
        release.wait(10)
        return []

    _fast_pipeline(monkeypatch, server, fit=blocking_fit)
    client = server.app.test_client()
    first, second = _demo_job(client), _demo_job(client)

    running = client.get(f"/api/run/{first}", buffered=False)  # worker blocks in fit
    try:
        busy = client.get(f"/api/run/{second}")
        assert busy.status_code == 503
        assert busy.get_json()["error"] == "busy, try again in a minute"
        assert busy.headers["Retry-After"] == "60"
    finally:
        release.set()
    running.get_data()
    running.close()

    # the slot is free again once the first job has finished
    assert client.get(f"/api/run/{second}").status_code == 200


def test_upload_rate_limit_per_ip(load_server):
    server = load_server()
    client = server.app.test_client()
    empty = {"model": "degraded_coating"}  # cheap: answered 400 "no files"

    for _ in range(10):
        assert client.post("/api/upload", data=empty, headers=API_HEADERS).status_code == 400
    limited = client.post("/api/upload", data=empty, headers=API_HEADERS)
    assert limited.status_code == 429
    assert "try again in a minute" in limited.get_json()["error"]
    assert limited.headers["Retry-After"] == "60"

    # X-Forwarded-For is ignored without EISPIPE_TRUST_PROXY ...
    spoofed = {**API_HEADERS, "X-Forwarded-For": "198.51.100.7"}
    assert client.post("/api/upload", data=empty, headers=spoofed).status_code == 429
    # ... while a genuinely different client address has its own budget
    other = client.post(
        "/api/upload", data=empty, headers=API_HEADERS,
        environ_overrides={"REMOTE_ADDR": "10.0.0.9"},
    )
    assert other.status_code == 400


def test_forwarded_for_trusted_only_with_trust_proxy(load_server):
    server = load_server(EISPIPE_TRUST_PROXY="1")
    client = server.app.test_client()
    empty = {"model": "degraded_coating"}

    # all from the proxy's address, but different forwarded clients
    for i in range(12):
        headers = {**API_HEADERS, "X-Forwarded-For": f"203.0.113.{i}"}
        assert client.post("/api/upload", data=empty, headers=headers).status_code == 400

    same = {**API_HEADERS, "X-Forwarded-For": "203.0.113.200"}
    for _ in range(10):
        assert client.post("/api/upload", data=empty, headers=same).status_code == 400
    assert client.post("/api/upload", data=empty, headers=same).status_code == 429


def test_upload_size_and_file_count_from_env(load_server):
    server = load_server(EISPIPE_MAX_UPLOAD_MB="1", EISPIPE_MAX_FILES="2")
    client = server.app.test_client()

    assert server.app.config["MAX_CONTENT_LENGTH"] == 1024 * 1024
    big = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"a" * (1024 * 1024 + 1)), "big.csv")},
        headers=API_HEADERS,
    )
    assert big.status_code == 413
    assert "1 MB" in big.get_json()["error"]

    files = [(io.BytesIO(b"x"), f"f{i}.csv") for i in range(3)]
    many = client.post(
        "/api/upload",
        data={"manifest": (io.BytesIO(b"filename,immersion_time_h\n"), "m.csv"), "file": files},
        headers=API_HEADERS,
    )
    assert many.status_code == 400
    assert "too many files (max 2)" in many.get_json()["error"]


def test_job_timeout_reports_failed_stage(load_server, monkeypatch):
    server = load_server(EISPIPE_JOB_TIMEOUT_S="0.5")

    def slow_fit(*args, **kwargs):
        time.sleep(3)
        return []

    _fast_pipeline(monkeypatch, server, fit=slow_fit)
    client = server.app.test_client()
    job_id = _demo_job(client)

    start = time.monotonic()
    stream = client.get(f"/api/run/{job_id}").get_data(as_text=True)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # the stream does not wait for the slow stage
    final = _events(stream)[-1]
    assert final["stage"] == "fit"
    assert final["status"] == "failed"
    assert "time limit" in final["message"]
    assert job_id not in server.jobs


# ---------------------------------------------------------------------------
# streaming through proxies
# ---------------------------------------------------------------------------


def test_sse_response_headers(load_server, monkeypatch):
    server = load_server()
    _fast_pipeline(monkeypatch, server)
    client = server.app.test_client()

    r = client.get(f"/api/run/{_demo_job(client)}")

    assert r.headers["Content-Type"].startswith("text/event-stream")
    assert r.headers["Cache-Control"] == "no-cache"
    assert r.headers["X-Accel-Buffering"] == "no"


def test_heartbeat_comment_while_a_stage_runs(load_server, monkeypatch):
    server = load_server()

    def slow_fit(*args, **kwargs):
        time.sleep(0.5)
        return []

    _fast_pipeline(monkeypatch, server, fit=slow_fit)
    monkeypatch.setattr(server, "HEARTBEAT_S", 0.05)
    client = server.app.test_client()

    stream = client.get(f"/api/run/{_demo_job(client)}").get_data(as_text=True)

    assert ": heartbeat\n\n" in stream
    assert _events(stream)[-1]["stage"] == "plot"  # the run still finished


# ---------------------------------------------------------------------------
# security headers
# ---------------------------------------------------------------------------


def _csp(response) -> dict[str, str]:
    policy = response.headers["Content-Security-Policy"]
    return {
        part.strip().split(" ", 1)[0]: part.strip().partition(" ")[2]
        for part in policy.split(";")
        if part.strip()
    }


@pytest.mark.parametrize(
    "method, path, headers",
    [
        ("GET", "/", {}),
        ("GET", "/healthz", {}),
        ("GET", "/api/run/unknown", {}),
        ("GET", "/server.py", {}),
        ("POST", "/api/upload", {}),  # 403: missing X-Requested-With
    ],
)
def test_security_headers_on_every_response(load_server, method, path, headers):
    server = load_server()
    client = server.app.test_client()

    r = client.open(path, method=method, headers=headers)

    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["X-Frame-Options"] == "DENY"
    csp = _csp(r)
    assert csp["img-src"] == "'self' data:"
    assert csp["script-src"].startswith("'self'")
    assert csp["style-src"].startswith("'self'")
    assert "unsafe-inline" not in r.headers["Content-Security-Policy"]


def test_index_inline_script_and_style_carry_a_per_request_nonce(load_server):
    server = load_server()
    client = server.app.test_client()

    nonces = []
    for _ in range(2):
        r = client.get("/")
        csp = _csp(r)
        nonce = re.fullmatch(r"'self' 'nonce-([A-Za-z0-9_-]+)'", csp["script-src"]).group(1)
        assert csp["style-src"] == f"'self' 'nonce-{nonce}'"
        body = r.get_data(as_text=True)
        assert f'<script nonce="{nonce}">' in body
        assert f'<style nonce="{nonce}">' in body
        assert "<script>" not in body and "<style>" not in body
        nonces.append(nonce)

    assert nonces[0] != nonces[1]
