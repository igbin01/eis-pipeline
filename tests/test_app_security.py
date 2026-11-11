"""Security regression tests for the Flask web app (app/server.py).

Every test points ``tempfile`` at a private directory under pytest's
``tmp_path`` so job temp dirs -- and anything that escapes them -- stay inside
the test sandbox.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import pytest

from helpers import synth_coating

SERVER_PATH = Path(__file__).resolve().parents[1] / "app" / "server.py"


@pytest.fixture(scope="module")
def server():
    name = "eis_app_server_security"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Flask derives its root_path from sys.modules[import_name]; without this it
    # falls back to the cwd and static/index lookups go to the wrong directory.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


@pytest.fixture
def job_base(tmp_path, monkeypatch):
    """Nested temp base so ``../`` from a job dir still lands inside tmp_path."""
    base = tmp_path / "a" / "b" / "c"
    base.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    return base


def _job_dirs(base: Path) -> list[Path]:
    return [p for p in base.iterdir() if p.is_dir() and p.name.startswith("eispipe_job_")]


def _files_outside_job_dirs(root: Path, base: Path) -> list[Path]:
    jobs = _job_dirs(base)
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(p.is_relative_to(j) for j in jobs)
    ]


def _csv_bytes(n: int = 30) -> bytes:
    freq, Z = synth_coating(n=n)
    lines = ["freq,Zreal,Zimag"]
    lines += [f"{float(f)!r},{float(z.real)!r},{float(z.imag)!r}" for f, z in zip(freq, Z)]
    return ("\n".join(lines) + "\n").encode()


def _post(client, data):
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


# ---------------------------------------------------------------------------
# (a) uploaded filenames must not escape the job temp dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_single_upload_filename_cannot_escape_job_dir(client, job_base, tmp_path, kind):
    name = "../../escape.csv" if kind == "relative" else str(tmp_path / "abs_escape.csv")

    _post(client, {"file": (io.BytesIO(_csv_bytes()), name)})

    assert _files_outside_job_dirs(tmp_path, job_base) == []


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_manifest_upload_filenames_cannot_escape_job_dir(client, job_base, tmp_path, kind):
    if kind == "relative":
        manifest_name, data_name = "../../m_escape.csv", "../../d_escape.csv"
    else:
        manifest_name = str(tmp_path / "m_abs_escape.csv")
        data_name = str(tmp_path / "d_abs_escape.csv")
    manifest = f"filename,immersion_time_h\n{data_name},0\n".encode()

    _post(
        client,
        {
            "manifest": (io.BytesIO(manifest), manifest_name),
            "file": [(io.BytesIO(_csv_bytes()), data_name)],
        },
    )

    assert _files_outside_job_dirs(tmp_path, job_base) == []


# ---------------------------------------------------------------------------
# (b) manifest rows must be confined to the uploaded files
# ---------------------------------------------------------------------------

SECRET = "SECRET_MARKER_first_line_of_a_private_file"


@pytest.mark.parametrize("kind", ["absolute", "dotdot"])
def test_manifest_row_outside_upload_is_rejected_without_leaking(client, job_base, kind):
    secret = job_base / "secret.csv"  # sibling of the job dirs: one "../" away
    secret.write_text(SECRET + "\nx\n")
    row = str(secret) if kind == "absolute" else "../secret.csv"
    manifest = f"filename,immersion_time_h\n{row},0\n".encode()

    r = _post(
        client,
        {
            "manifest": (io.BytesIO(manifest), "manifest.csv"),
            "file": [(io.BytesIO(_csv_bytes()), "a.csv")],
        },
    )

    body = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "error" in r.get_json()
    assert SECRET not in body
    assert str(job_base) not in body


def test_valid_manifest_upload_still_loads(client, job_base, server):
    manifest = b"filename,immersion_time_h,sample_id\nt0.csv,0,P1\nt24.csv,24,P1\n"

    r = _post(
        client,
        {
            "manifest": (io.BytesIO(manifest), "manifest.csv"),
            "file": [
                (io.BytesIO(_csv_bytes()), "t0.csv"),
                (io.BytesIO(_csv_bytes()), "t24.csv"),
            ],
        },
    )

    assert r.status_code == 200, r.get_data(as_text=True)
    series = server.jobs[r.get_json()["job_id"]]["series"]
    assert list(series.times) == [0.0, 24.0]
    assert series.sample_id == "P1"


def test_valid_single_upload_still_loads(client, job_base, server):
    r = _post(
        client,
        {"file": (io.BytesIO(_csv_bytes()), "panel.csv"), "immersion_time_h": "12"},
    )

    assert r.status_code == 200, r.get_data(as_text=True)
    series = server.jobs[r.get_json()["job_id"]]["series"]
    assert list(series.times) == [12.0]


# ---------------------------------------------------------------------------
# (c) only the page is served, not the app/ directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/server.py", "/__init__.py"])
def test_app_source_files_are_not_served(client, path):
    assert client.get(path).status_code == 404


def test_index_page_is_still_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>EIS Pipeline</title>" in r.get_data(as_text=True)


# ---------------------------------------------------------------------------
# (d) error responses must not leak temp paths
# ---------------------------------------------------------------------------


def test_upload_error_does_not_leak_temp_path(client, job_base):
    # manifest names a file that was not uploaded -> loader fails on a temp path
    manifest = b"filename,immersion_time_h\nmissing.csv,0\n"

    r = _post(
        client,
        {
            "manifest": (io.BytesIO(manifest), "manifest.csv"),
            "file": [(io.BytesIO(_csv_bytes()), "a.csv")],
        },
    )

    assert r.status_code == 400
    assert str(job_base) not in r.get_data(as_text=True)


@pytest.mark.parametrize(
    "failing", ["validate_series", "fit_series", "metrics_table", "plot_metric"]
)
def test_stream_errors_do_not_leak_exception_text(client, job_base, server, monkeypatch, failing):
    import pandas as pd

    leak = f"{job_base}/eispipe_job_x/private.csv"

    def boom(*args, **kwargs):
        raise RuntimeError(f"failed reading {leak}")

    # Short-circuit the expensive stages so each stage's error path is reached.
    monkeypatch.setattr(server, "fit_series", lambda *a, **k: [])
    monkeypatch.setattr(server, "metrics_table", lambda fits: pd.DataFrame())
    monkeypatch.setattr(server, failing, boom)

    r = _post(client, {"use_demo_data": "1"})
    assert r.status_code == 200
    stream = client.get(f"/api/run/{r.get_json()['job_id']}").get_data(as_text=True)

    assert '"status": "failed"' in stream
    assert leak not in stream
    assert str(job_base) not in stream
