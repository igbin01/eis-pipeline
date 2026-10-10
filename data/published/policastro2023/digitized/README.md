# Digitized Figure 5c,d — not yet done

Placeholder for points digitized from Policastro et al. 2023, Figure 5c
(Bode magnitude) and Figure 5d (Bode phase), for the same 0.01 M NaCl, 5 degC
condition as `../table2_values.csv`. See `../SOURCE.md`.

Digitize with [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/), one
CSV per immersion time, expected columns:

```
freq_hz, Z_mod_ohm, phase_deg
```

Immersion times to digitize (h): 0, 168, 336, 504, 840, 1008, 1176, 1344,
1512, 1680, 3192 (matching Table 2).

Suggested filename pattern: `bode_<time_h>h.csv` (e.g. `bode_0h.csv`,
`bode_3192h.csv`).
