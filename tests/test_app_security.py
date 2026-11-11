"""Security regression tests for the Flask web app (app/server.py).

Every test points ``tempfile`` at a private directory under pytest's
``tmp_path`` so job temp dirs -- and anything that escapes them -- stay inside
the test sandbox.
"""

from __future__ import annotations

import importlib.util
import io
import os
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
    # The module reads EISPIPE_* at import; load it with the local defaults.
    saved = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("EISPIPE_")]}
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        os.environ.update(saved)
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _fresh_upload_rate_limit(server):
    """This module shares one server across many tests; reset the per-IP limit."""
    server._upload_log.clear()


ALLOWED_BASE_URL = "http://localhost:5050"


@pytest.fixture
def bare_client(server):
    """Test client on an allowed Host, without the X-Requested-With header."""
    from flask.testing import FlaskClient

    class _LocalClient(FlaskClient):
        def open(self, *args, **kwargs):
            kwargs.setdefault("base_url", ALLOWED_BASE_URL)
            return super().open(*args, **kwargs)

    server.app.config["TESTING"] = True
    server.app.test_client_class = _LocalClient
    return server.app.test_client()


@pytest.fixture
def client(bare_client):
    """What the browser UI sends: allowed Host plus X-Requested-With."""
    bare_client.environ_base["HTTP_X_REQUESTED_WITH"] = "eispipe"
    return bare_client


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
    lines += [
        f"{float(f)!r},{float(z.real)!r},{float(z.imag)!r}"
        for f, z in zip(freq, Z, strict=True)
    ]
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


# ---------------------------------------------------------------------------
# B4: upload size / part / file-count limits
# ---------------------------------------------------------------------------

LIMIT_25_MB = 25 * 1024 * 1024
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "failure_case"


def test_oversized_upload_is_rejected_with_413(client, job_base, server):
    assert server.app.config["MAX_CONTENT_LENGTH"] == LIMIT_25_MB

    r = _post(client, {"file": (io.BytesIO(b"a" * (LIMIT_25_MB + 1)), "big.csv")})

    assert r.status_code == 413
    assert "too large" in r.get_json()["error"]


def test_too_many_form_parts_is_rejected_with_413(client, job_base, server):
    limit = server.app.config["MAX_FORM_PARTS"]
    assert limit <= 100

    data = {f"field{i}": "x" for i in range(limit + 1)}
    r = _post(client, data)

    assert r.status_code == 413
    assert "error" in r.get_json()


def test_too_many_spectrum_files_is_rejected(client, job_base):
    names = [f"f{i}.csv" for i in range(65)]
    manifest = "filename,immersion_time_h\n" + "".join(f"{n},{i}\n" for i, n in enumerate(names))

    r = _post(
        client,
        {
            "manifest": (io.BytesIO(manifest.encode()), "manifest.csv"),
            "file": [(io.BytesIO(_csv_bytes(8)), n) for n in names],
        },
    )

    assert r.status_code == 400
    assert "too many files" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# B5: temp dirs and jobs are cleaned up
# ---------------------------------------------------------------------------


def _upload_case(kind: str) -> dict:
    if kind == "load_error":
        return {"file": (io.BytesIO(b"not,a,spectrum\n1,2,3\n"), "bad.csv")}
    if kind == "rejected_manifest":
        return {
            "manifest": (io.BytesIO(b"filename,immersion_time_h\n../x.csv,0\n"), "manifest.csv"),
            "file": [(io.BytesIO(_csv_bytes()), "a.csv")],
        }
    if kind == "no_files":
        return {"model": "degraded_coating"}
    if kind == "demo":
        return {"use_demo_data": "1"}
    if kind == "valid_single":
        return {"file": (io.BytesIO(_csv_bytes()), "ok.csv"), "immersion_time_h": "0"}
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["load_error", "rejected_manifest", "no_files", "demo", "valid_single"]
)
def test_no_job_temp_dir_remains_after_upload(client, job_base, kind):
    _post(client, _upload_case(kind))

    assert _job_dirs(job_base) == []


def test_finished_job_leaves_no_temp_dir_and_is_dropped(client, job_base, server):
    r = _post(client, _upload_case("valid_single"))
    job_id = r.get_json()["job_id"]

    stream = client.get(f"/api/run/{job_id}").get_data(as_text=True)

    assert '"job complete"' in stream
    assert job_id not in server.jobs
    assert _job_dirs(job_base) == []


def test_abandoned_job_expires(client, job_base, server):
    job_id = _post(client, {"use_demo_data": "1"}).get_json()["job_id"]
    server.jobs[job_id]["created"] -= server.JOB_TTL_S + 1

    r = client.get(f"/api/run/{job_id}")

    assert r.status_code == 404
    assert job_id not in server.jobs


# ---------------------------------------------------------------------------
# B7: Host / Origin / X-Requested-With checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1:5050", "localhost:5050"])
def test_allowed_hosts_are_accepted(client, job_base, host):
    r = client.post(
        "/api/upload", data={"use_demo_data": "1"}, headers={"Host": host}
    )
    assert r.status_code == 200


@pytest.mark.parametrize("host", ["evil.example:5050", "localhost", "127.0.0.1:8080"])
def test_foreign_host_is_rejected_on_api(client, job_base, host):
    job_id = _post(client, {"use_demo_data": "1"}).get_json()["job_id"]

    post = client.post("/api/upload", data={"use_demo_data": "1"}, headers={"Host": host})
    get = client.get(f"/api/run/{job_id}", headers={"Host": host})

    assert post.status_code == 403
    assert get.status_code == 403


@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://localhost:8080"])
def test_foreign_origin_post_is_rejected(client, job_base, origin):
    r = client.post("/api/upload", data={"use_demo_data": "1"}, headers={"Origin": origin})
    assert r.status_code == 403


@pytest.mark.parametrize("origin", ["http://127.0.0.1:5050", "http://localhost:5050"])
def test_same_origin_post_is_accepted(client, job_base, origin):
    r = client.post("/api/upload", data={"use_demo_data": "1"}, headers={"Origin": origin})
    assert r.status_code == 200


def test_upload_without_custom_header_is_rejected(bare_client, job_base):
    r = bare_client.post("/api/upload", data={"use_demo_data": "1"})
    assert r.status_code == 403


def test_index_page_sends_custom_header_on_uploads():
    html = (SERVER_PATH.parent / "index.html").read_text()
    assert html.count('"X-Requested-With": "eispipe"') == html.count('fetch("/api/upload"') == 2


# ---------------------------------------------------------------------------
# B6 follow-up: our own loader/validation errors come back specific but safe
# ---------------------------------------------------------------------------


def test_missing_freq_column_gives_specific_safe_message(client, job_base):
    csv = b"foo,Zreal,Zimag\n1,2,-3\n2,3,-4\n"

    r = _post(client, {"file": (io.BytesIO(csv), "nofreq.csv"), "immersion_time_h": "0"})

    body = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "could not find a 'freq' column" in r.get_json()["error"]
    assert str(job_base) not in body


def test_bad_manifest_gives_specific_safe_message(client, job_base):
    r = _post(
        client,
        {
            "manifest": (io.BytesIO(b"filename,sample_id\na.csv,P1\n"), "manifest.csv"),
            "file": [(io.BytesIO(_csv_bytes()), "a.csv")],
        },
    )

    body = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "missing required column" in r.get_json()["error"]
    assert str(job_base) not in body


# ---------------------------------------------------------------------------
# normal flows still work end to end
# ---------------------------------------------------------------------------


def _events(stream: str) -> list[dict]:
    import json

    return [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]


def test_demo_run_end_to_end(client, job_base, server):
    job_id = _post(client, {"use_demo_data": "1"}).get_json()["job_id"]

    events = _events(client.get(f"/api/run/{job_id}").get_data(as_text=True))

    assert [e for e in events if e["status"] == "failed"] == []
    final = events[-1]
    assert final["stage"] == "plot" and final["message"] == "job complete"
    assert len(final["metrics_table"]) == 6
    assert final["bode_image_png_base64"] and final["rpo_vs_time_image_png_base64"]
    assert job_id not in server.jobs
    assert _job_dirs(job_base) == []


def test_failure_case_fixture_end_to_end(client, job_base, server):
    names = ["clean_spectrum.csv", "zero_impedance_spectrum.csv"]
    r = _post(
        client,
        {
            "manifest": (io.BytesIO((FIXTURES / "manifest.csv").read_bytes()), "manifest.csv"),
            "file": [(io.BytesIO((FIXTURES / n).read_bytes()), n) for n in names],
        },
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    job_id = r.get_json()["job_id"]

    events = _events(client.get(f"/api/run/{job_id}").get_data(as_text=True))

    final = events[-1]
    assert final["message"] == "job complete"
    assert [k["passed"] for k in final["kk_pass_fail"]] == [True, False]
    assert [m["success"] for m in final["metrics_table"]] == [True, False]
    fit_failures = [e for e in events if e["stage"] == "fit" and e["status"] == "failed"]
    assert [e["index"] for e in fit_failures] == [1]
    assert job_id not in server.jobs
    assert _job_dirs(job_base) == []
