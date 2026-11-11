from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from werkzeug.utils import secure_filename

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# eispipe is imported after the sys.path setup above so the app also runs from
# a source checkout where the package is not installed.
from eispipe.fit import fit_series  # noqa: E402
from eispipe.io import load_csv, load_manifest, load_series  # noqa: E402
from eispipe.metrics import metrics_table  # noqa: E402
from eispipe.plots import bode, plot_metric  # noqa: E402
from eispipe.spectrum import Spectrum, SpectrumSeries  # noqa: E402
from eispipe.validate import validate_series  # noqa: E402

APP_DIR = Path(__file__).resolve().parent

# No static folder: only the page itself is served, never the app/ directory.
app = Flask(__name__, static_folder=None)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SPECTRUM_FILES = 64
JOB_TTL_S = 30 * 60  # jobs never streamed within this time are dropped
ALLOWED_HOSTS = frozenset({"127.0.0.1:5050", "localhost:5050"})
ALLOWED_ORIGINS = frozenset(f"http://{host}" for host in ALLOWED_HOSTS)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["MAX_FORM_PARTS"] = MAX_SPECTRUM_FILES + 16  # spectra + manifest + form fields

jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


@app.before_request
def _check_request_origin():
    """Only accept API calls addressed to this local server from its own page."""
    if not request.path.startswith("/api/"):
        return None
    if request.host not in ALLOWED_HOSTS:
        return jsonify({"error": "forbidden host"}), 403
    origin = request.headers.get("Origin")
    if request.method == "POST" and origin is not None and origin not in ALLOWED_ORIGINS:
        return jsonify({"error": "forbidden origin"}), 403
    if request.path == "/api/upload" and request.headers.get("X-Requested-With") != "eispipe":
        return jsonify({"error": "missing X-Requested-With: eispipe header"}), 403
    return None


@app.errorhandler(413)
def _upload_too_large(_exc):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify(
        {
            "error": f"upload too large (limit {limit_mb} MB and "
            f"{app.config['MAX_FORM_PARTS']} form parts)"
        }
    ), 413


def _json_safe(value: Any) -> Any:
    """Recursively coerce payloads into JSON-safe values for the SSE stream.

    Browser parsers and strict JSON should never see NaN/Inf or other
    non-finite floats as raw tokens. We map them to None so the frontend
    receives a well-formed object and can keep the failure row visible
    instead of dropping the whole event.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


@app.get("/")
def index():
    return send_file(APP_DIR / "index.html")


def _emit_event(stage: str, status: str, **extra: Any) -> str:
    payload = {"stage": stage, "status": status}
    payload.update(extra)
    payload = _json_safe(payload)
    serialized = json.dumps(payload, allow_nan=False)
    return f"event: update\ndata: {serialized}\n\n"


def _register_job(job: dict[str, Any]) -> None:
    with _jobs_lock:
        jobs[job["job_id"]] = job


def _expire_jobs() -> None:
    cutoff = time.monotonic() - JOB_TTL_S
    with _jobs_lock:
        for job_id in [j for j, job in jobs.items() if job["created"] < cutoff]:
            del jobs[job_id]


def _strip_temp_paths(message: str, tmpdir: Path) -> str:
    """Remove the job temp dir from *message*, leaving bare file names."""
    for prefix in {str(tmpdir.resolve()), str(tmpdir)}:
        message = message.replace(prefix + os.sep, "").replace(prefix, "")
    return message


class _UploadRejected(ValueError):
    """Upload refused for a reason whose message is safe to show the client."""


def _save_upload(uploaded, tmpdir: Path, stem: str) -> Path:
    """Save under a server-chosen name in *tmpdir*; only the extension is kept."""
    safe = secure_filename(uploaded.filename or "")
    if not safe:
        raise _UploadRejected("invalid filename")
    dest = tmpdir / f"{stem}{Path(safe).suffix.lower()}"
    uploaded.save(dest)
    return dest


def _resolve_manifest_row(name: str, stored: dict[str, Path], tmpdir: Path) -> str:
    """Map a manifest ``filename`` to its stored upload; reject anything else."""
    if name != Path(name).name or "\\" in name or name not in stored:
        raise _UploadRejected("manifest references a file that was not uploaded")
    target = stored[name].resolve()
    if target.parent != tmpdir.resolve():
        raise _UploadRejected("manifest references a file that was not uploaded")
    return target.name


@app.post("/api/upload")
def upload():
    model = request.form.get("model") or "degraded_coating"
    use_demo = str(request.form.get("use_demo_data", "")).lower() in {"1", "true", "yes", "on"}
    if not use_demo:
        use_demo = str(request.form.get("use_demo", "")).lower() in {"1", "true", "yes", "on"}

    _expire_jobs()
    job_id = uuid.uuid4().hex
    tmpdir = Path(tempfile.mkdtemp(prefix="eispipe_job_"))
    job = {"job_id": job_id, "model": model, "created": time.monotonic(), "series": None}

    try:
        if use_demo:
            from demo_run import IMMERSION_TIMES_H, simulate_spectrum

            spectra = [
                simulate_spectrum(float(t), seed=i + 1)
                for i, t in enumerate(IMMERSION_TIMES_H)
            ]
            job["series"] = SpectrumSeries(spectra)
            _register_job(job)
            return jsonify({"job_id": job_id})

        files = request.files.getlist("file") or request.files.getlist("files") or []
        manifest_file = request.files.get("manifest") or request.files.get("manifest_csv")
        if len(files) > MAX_SPECTRUM_FILES:
            return jsonify({"error": f"too many files (max {MAX_SPECTRUM_FILES})"}), 400

        if manifest_file:
            manifest_path = _save_upload(manifest_file, tmpdir, "manifest")
            stored: dict[str, Path] = {}
            for i, uploaded in enumerate(files):
                if uploaded.filename:
                    stored[uploaded.filename] = _save_upload(uploaded, tmpdir, f"data_{i}")
            manifest = load_manifest(manifest_path)
            manifest["filename"] = [
                _resolve_manifest_row(str(name), stored, tmpdir) for name in manifest["filename"]
            ]
            series = load_series(manifest, data_dir=tmpdir)
            job["series"] = series
            _register_job(job)
            return jsonify({"job_id": job_id})

        if len(files) == 1:
            csv_path = _save_upload(files[0], tmpdir, "data_0")
            meta = {}
            sample_id = request.form.get("sample_id")
            immersion_time_h = request.form.get("immersion_time_h")
            area_cm2 = request.form.get("area_cm2")
            if sample_id:
                meta["sample_id"] = sample_id
            if immersion_time_h is not None:
                meta["immersion_time_h"] = float(immersion_time_h)
            if area_cm2 is not None:
                meta["area_cm2"] = float(area_cm2)
            spec = load_csv(csv_path, **meta)
            job["series"] = SpectrumSeries([spec])
            _register_job(job)
            return jsonify({"job_id": job_id})

        return jsonify({"error": "upload requires either one CSV or a manifest CSV + files, or the demo flag"}), 400
    except ValueError as exc:
        # Our own loader/validation messages (incl. _UploadRejected): specific
        # but with the temp dir stripped.
        return jsonify({"error": _strip_temp_paths(str(exc), tmpdir)}), 400
    except Exception:
        # Intentionally broad: any unexpected failure becomes a generic 400;
        # details go to the server log only.
        app.logger.exception("upload %s failed", job_id)
        return jsonify({"error": "could not load the uploaded files"}), 400
    finally:
        # Spectra are fully in memory by now; the uploaded files are not needed.
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/run/<job_id>")
def run_job(job_id: str):
    _expire_jobs()
    with _jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    def _stream():
        series_obj = job["series"]
        model = job["model"]

        # load
        yield _emit_event("load", "in_progress", message="loading spectra")
        if series_obj is None:
            yield _emit_event("load", "failed", message="no series was loaded")
            return
        yield _emit_event("load", "done", message=f"loaded {len(series_obj)} spectra")

        # validate
        yield _emit_event("validate", "in_progress", message="running KK validation")
        kk_records = []
        try:
            kk_df = validate_series(series_obj)
            kk_records = kk_df.to_dict(orient="records")
            yield _emit_event("validate", "done", message="KK validation complete", kk_pass_fail=kk_records)
        except Exception:
            # Intentionally broad (all stages): report a failed stage to the
            # client instead of breaking the event stream; details are logged.
            app.logger.exception("job %s: validation failed", job_id)
            yield _emit_event("validate", "failed", message="KK validation failed")

        # fit
        yield _emit_event("fit", "in_progress", message="fitting spectra")
        fit_events = []

        def on_progress(index: int, total: int, result: Any) -> None:
            if result.success:
                fit_events.append(
                    {
                        "stage": "fit",
                        "status": "done",
                        "index": index,
                        "total": total,
                        "message": "fit complete",
                        "immersion_time_h": result.immersion_time_h,
                    }
                )
            else:
                fit_events.append(
                    {
                        "stage": "fit",
                        "status": "failed",
                        "index": index,
                        "total": total,
                        "message": result.message,
                        "immersion_time_h": result.immersion_time_h,
                    }
                )

        try:
            fits = fit_series(series_obj, model, validate=True, on_progress=on_progress)
        except Exception:  # intentionally broad: see the validation stage
            app.logger.exception("job %s: fitting failed", job_id)
            yield _emit_event("fit", "failed", message="fitting failed")
            return

        for event in fit_events:
            message = event.get("message", "")
            status = event.get("status")
            idx = event.get("index")
            total = event.get("total")
            if status == "failed":
                yield _emit_event("fit", "failed", message=message, index=idx, total=total, immersion_time_h=event.get("immersion_time_h"))
            else:
                yield _emit_event("fit", "done", message=message, index=idx, total=total, immersion_time_h=event.get("immersion_time_h"))

        yield _emit_event("fit", "done", message=f"fit complete for {len(fits)} spectra")

        # metrics
        yield _emit_event("metrics", "in_progress", message="computing metrics")
        try:
            metrics = metrics_table(fits)
            metrics_records = metrics.to_dict(orient="records")
        except Exception:  # intentionally broad: see the validation stage
            app.logger.exception("job %s: metrics failed", job_id)
            yield _emit_event("metrics", "failed", message="metrics computation failed")
            return
        yield _emit_event("metrics", "done", message="metrics computed", metrics_table=metrics_records)

        # plot
        yield _emit_event("plot", "in_progress", message="building plots")
        try:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
            for fit_result in fits:
                if fit_result.success:
                    spec = Spectrum(freq=fit_result.freq, Z=fit_result.Z_data, immersion_time_h=fit_result.immersion_time_h)
                    bode(spec, fit=fit_result, axes=axes)
            fig.tight_layout()
            bode_buf = io.BytesIO()
            fig.savefig(bode_buf, format="png", dpi=150)
            plt.close(fig)

            plot_ax = plot_metric(metrics, "R_po", logy=True)
            plot_ax.set_title("R_po vs. immersion time")
            plot_ax.figure.tight_layout()
            rpo_buf = io.BytesIO()
            plot_ax.figure.savefig(rpo_buf, format="png", dpi=150)
            plt.close(plot_ax.figure)

            bode_png = base64.b64encode(bode_buf.getvalue()).decode("ascii")
            rpo_png = base64.b64encode(rpo_buf.getvalue()).decode("ascii")
        except Exception:  # intentionally broad: see the validation stage
            app.logger.exception("job %s: plotting failed", job_id)
            yield _emit_event("plot", "failed", message="plotting failed")
            return

        yield _emit_event(
            "plot",
            "done",
            message="job complete",
            metrics_table=metrics_records,
            kk_pass_fail=kk_records,
            bode_image_png_base64=bode_png,
            rpo_vs_time_image_png_base64=rpo_png,
        )

    def _stream_then_drop_job():
        try:
            yield from _stream()
        finally:
            with _jobs_lock:
                jobs.pop(job_id, None)

    return Response(stream_with_context(_stream_then_drop_job()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
