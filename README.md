# eispipe

eispipe analyses electrochemical impedance spectroscopy (EIS) data from organic
coatings over a series of immersion times. It loads impedance sweeps, checks
them with a Kramers-Kronig test, fits an equivalent-circuit model to each one,
and turns the fitted parameters into coating metrics: coating capacitance
(Hsu-Mansfeld), water uptake (Brasher-Kingsbury), pore resistance, and
breakpoint frequency. The output shows how the coating degrades over time.

## Pipeline

| Step | Module | Main entry points |
|---|---|---|
| 1. **Load** | `eispipe.io` | `load_csv`, `load_instrument`, `load_series` (sweeps + manifest CSV giving immersion times), `crop_frequencies`, `drop_inductive` |
| 2. **Validate** | `eispipe.validate` | `kramers_kronig`, `validate_series` (Lin-KK test → `KKResult`) |
| 3. **Fit** | `eispipe.fit` | `fit_spectrum`, `fit_series` (→ `FitResult` / `FitSeries`) |
| 4. **Metrics** | `eispipe.metrics` | `cpe_to_capacitance`, `coating_capacitance`, `water_uptake_series`, `breakpoint_frequency`, `metrics_table` |
| 5. **Plot** | `eispipe.plots` | `nyquist`, `bode`, `nyquist_series`, `kk_residuals`, `plot_metric` |

Immersion times come from a sidecar manifest CSV, never from file timestamps.
Impedance is stored with capacitive reactance negative (`Z.imag < 0`), and
Nyquist plots show `-Z.imag`. Import `plots` explicitly with
`from eispipe import plots`. Importing the package alone does not load
matplotlib.

## Circuit models

Circuit strings use [impedance.py](https://impedancepy.readthedocs.io/) syntax.

| Name | Circuit string | Parameters |
|---|---|---|
| `intact_coating` | `R0-p(R1,CPE1)` | `R_e`, `R_po`, `Q_c`, `n_c` |
| `degraded_coating` | `R0-p(CPE1,R1-p(CPE2,R2))` | `R_e`, `Q_c`, `n_c`, `R_po`, `Q_dl`, `n_dl`, `R_ct` |

- **intact_coating**: a single time constant. Electrolyte resistance in series
  with the coating pore resistance, which is in parallel with the coating CPE.
  Use it while the film is still a good barrier.
- **degraded_coating**: two nested time constants. The substrate branch
  (charge-transfer resistance in parallel with the double-layer CPE) is in
  series with the pore resistance, and that whole branch is in parallel with
  the coating CPE.

Look up a model with `eispipe.get_model(name)`.

## Install and test

Requires Python ≥ 3.10.

```
pip install -e ".[dev]"
pytest
```

The `dev` extra installs pytest and Flask, which the web app needs.

## Web app

```
python app/server.py
```

Then open <http://127.0.0.1:5050>. You can upload a series or run the
built-in demo. The app streams the KK validation, fitting, and metrics
results, then displays Bode and R_po-vs-time plots.

## Validation

`scripts/validate_policastro2023.py` compares `cpe_to_capacitance` with
Table 2 of Policastro et al. 2023 (*Materials* 16, 3623,
doi:10.3390/ma16103623): an epoxy/polyurethane coating in 0.01 M NaCl at 5 °C,
with 11 immersion times from 0 to 3192 h. The input values are in
`data/published/policastro2023/table2_values.csv`.

Starting from the published R_po, Y0, and m, `cpe_to_capacitance` matches the
paper's coating capacitance to within ±5.3% on 10 of the 11 points. The worst
case is −8.1% at t = 0.

**This validates the capacitance formula only, not the fitting step.** The
script uses the paper's already-fitted parameters. It does not test whether
`fit_spectrum` would get the same R_po, Y0, and m from the raw spectra.

## Known limitations

- **Brasher-Kingsbury water uptake underestimates uptake.** The formula expects
  a dry-film reference capacitance. `water_uptake_series` and `metrics_table`
  use the earliest immersion spectrum as the reference by default
  (`reference="first"`), so any water absorbed before that first measurement is
  missed.
- **No real instrument export has been tested yet.** `load_instrument` reads
  Gamry, BioLogic, Autolab, and other vendor files through impedance.py's
  readers. However, it has only been exercised on synthetic data and CSV input,
  not on actual files from an instrument.
