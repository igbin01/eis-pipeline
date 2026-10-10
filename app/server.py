from __future__ import annotations

import base64
import io
import json
import math
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from flask import Flask, Response, jsonify, request, stream_with_context

from eispipe.fit import fit_series
from eispipe.io import load_csv, load_series
from eispipe.metrics import metrics_table
from eispipe.plots import bode, plot_metric
from eispipe.spectrum import Spectrum, SpectrumSeries
from eispipe.validate import validate_series

app = Flask(__name__, static_folder=".", static_url_path="")

jobs: dict[str, dict[str, Any]] = {}


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
    return app.send_static_file("index.html")


def _emit_event(stage: str, status: str, **extra: Any) -> str:
    payload = {"stage": stage, "status": status}
    payload.update(extra)
    payload = _json_safe(payload)
    serialized = json.dumps(payload, allow_nan=False)
    return f"event: update\ndata: {serialized}\n\n"


def _cleanup_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if job:
        tmpdir = job.get("tmpdir")
        if isinstance(tmpdir, str):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
        jobs.pop(job_id, None)


@app.post("/api/upload")
def upload():
    model = request.form.get("model") or "degraded_coating"
    use_demo = str(request.form.get("use_demo_data", "")).lower() in {"1", "true", "yes", "on"}
    if not use_demo:
        use_demo = str(request.form.get("use_demo", "")).lower() in {"1", "true", "yes", "on"}

    job_id = uuid.uuid4().hex
    tmpdir = Path(tempfile.mkdtemp(prefix="eispipe_job_"))
    job = {"job_id": job_id, "model": model, "tmpdir": str(tmpdir), "series": None}

    try:
        if use_demo:
            from demo_run import IMMERSION_TIMES_H, simulate_spectrum

            spectra = [
                simulate_spectrum(float(t), seed=i + 1)
                for i, t in enumerate(IMMERSION_TIMES_H)
            ]
            job["series"] = SpectrumSeries(spectra)
            jobs[job_id] = job
            return jsonify({"job_id": job_id})

        files = request.files.getlist("file") or request.files.getlist("files") or []
        manifest_file = request.files.get("manifest") or request.files.get("manifest_csv")

        if manifest_file:
            manifest_path = tmpdir / manifest_file.filename
            manifest_file.save(manifest_path)
            for uploaded in files:
                if uploaded.filename:
                    dest = tmpdir / uploaded.filename
                    uploaded.save(dest)
            series = load_series(manifest_path, data_dir=tmpdir)
            job["series"] = series
            jobs[job_id] = job
            return jsonify({"job_id": job_id})

        if len(files) == 1:
            uploaded = files[0]
            csv_path = tmpdir / uploaded.filename
            uploaded.save(csv_path)
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
            jobs[job_id] = job
            return jsonify({"job_id": job_id})

        return jsonify({"error": "upload requires either one CSV or a manifest CSV + files, or the demo flag"}), 400
    except Exception as exc:
        _cleanup_job(job_id)
        return jsonify({"error": str(exc)}), 400


@app.get("/api/run/<job_id>")
def run_job(job_id: str):
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
        except Exception as exc:
            yield _emit_event("validate", "failed", message=str(exc))

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
        except Exception as exc:
            yield _emit_event("fit", "failed", message=str(exc))
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
        except Exception as exc:
            yield _emit_event("metrics", "failed", message=str(exc))
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
        except Exception as exc:
            yield _emit_event("plot", "failed", message=str(exc))
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

    return Response(stream_with_context(_stream()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
