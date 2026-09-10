eis-pipeline/
├── CLAUDE.md
├── pyproject.toml     # src-layout package "eispipe"; deps + [dev] extra
├── requirements.txt   # full pinned env (machine-generated)
├── src/eispipe/
│   ├── spectrum.py    # Spectrum / SpectrumSeries: metadata-carrying data types
│   ├── io.py          # load CSV + instrument exports; crop / drop-inductive preprocessing
│   ├── validate.py    # Lin-KK (Kramers-Kronig) check → KKResult
│   ├── models.py      # coating equivalent-circuit registry (intact / degraded)
│   ├── fit.py         # circuit fitting + parameter/uncertainty extraction
│   ├── metrics.py     # Hsu-Mansfeld C, Brasher-Kingsbury water uptake, Rpo, f_break
│   └── plots.py       # Nyquist, Bode, KK residuals, metric time-series (not re-exported)
├── tests/             # pytest; one test module per src module
└── data/raw/          # gitignored

Sign convention: impedance stored capacitive-is-negative (Z.imag < 0), Nyquist plots -Z.imag.
Immersion time comes from a sidecar manifest CSV (never file mtime).
Deferred by decision: pipeline.py, cli.py (no orchestrator until the real workflow is known).

Never scan or search .venv/; it contains thousands of installed library files.
