FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/eispipe

# Package first, so code-only changes to app/ reuse the dependency layer.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[web]"

COPY app/ app/
# The web app's "Load demo data" button imports scripts/demo_run.py.
COPY scripts/demo_run.py scripts/demo_run.py

USER app
EXPOSE 7860

# One worker: job state lives in this process's memory. sh -c expands ${PORT};
# exec hands signals straight to gunicorn.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads 8 -b 0.0.0.0:${PORT:-7860} app.server:app"]
