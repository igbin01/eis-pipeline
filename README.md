# eispipe

EIS analysis pipeline for organic coating degradation over immersion time.

Load impedance sweeps (CSV or instrument exports), validate them with a
Kramers-Kronig test, fit equivalent-circuit models, and track coating condition
metrics (water uptake, pore resistance) across an immersion series.

## Install (development)

```
pip install -e ".[dev]"
pytest
```

## Status

Under construction. Implemented so far: `spectrum` (data types), `io`
(loading + preprocessing).
