# Digitized Figure 5c,d — not yet done

Points digitized from Policastro et al. 2023, Figure 5c (Bode magnitude) and
Figure 5d (Bode phase), for the 0.01 M NaCl, 5 degC condition (same as
`../table2_values.csv`). See `../SOURCE.md`. Read by
`scripts/validate_policastro_fig5.py`.

Digitize with [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).
Immersion times (h): 0, 840, 3192. Two files per time `t`:

| File | Columns | Units |
|---|---|---|
| `t{t}h_mag.csv` | `freq_hz, Z_mod_ohm` | Hz, raw ohms (NOT ohm*cm^2) |
| `t{t}h_phase.csv` | `freq_hz, phase_deg` | Hz, degrees, negative for capacitive |

So: `t0h_mag.csv`, `t0h_phase.csv`, `t840h_mag.csv`, `t840h_phase.csv`,
`t3192h_mag.csv`, `t3192h_phase.csv`. The magnitude and phase files may use
different frequency points; phase is interpolated onto the magnitude
frequencies.
