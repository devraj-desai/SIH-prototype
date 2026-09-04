# Subsurface Ocean Temperature Reconstruction — Prototype

Frontend-only Streamlit prototype for the "Satellite Embedding-Based Deep
Learning Framework" problem statement (North Indian Ocean, 5°N–30°N,
45°E–105°E, 0.25° / daily).

No ML/DL model is included — all outputs come from placeholder functions
in `model_interface.py`, ready to be swapped for real inference.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

The map always shows the reconstructed temperature field (smooth
filled-contour style, like satellite temperature maps) for the chosen
depth and date. Use the **Selection mode** toggle in the sidebar:

- **Point** → click anywhere on the map → set SST, SSS, SSH, uo, vo and
  date in the sidebar → click **Predict Profile** → see a
  temperature-vs-depth curve for the 15 standard depth levels (0–1000 m),
  with the depth-slider value starred on the curve, plus a values table.
- **Region (rectangle)** → use the rectangle drawing tool (top-left of the
  map) to drag out a box → click **Predict Region Stats** → see
  mean/min/max/std metrics at the selected depth, plus a mean profile
  with a shaded min–max band across all depths, plus a values table.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: map, inputs, point profile, region stats |
| `model_interface.py` | **The only file to edit to connect the real model.** Contains `predict_temperature_profile()`, `predict_region_field()`, `predict_region_volume()`, and `predict_region_profile_stats()` with mock logic and full docstrings describing expected inputs/outputs |
| `map_utils.py` | Frontend-only styling: turns a 2D field into the smooth colored map overlay + legend. No model logic here. |
| `requirements.txt` | Python dependencies |

## Connecting the real model

In `model_interface.py`, replace the body of:

- `predict_temperature_profile(lat, lon, sst, sss, ssh, uo, vo, date)` →
  return `{"depths": [...15 values...], "temperature": [...15 values...]}`
- `predict_region_volume(date, lat_grid, lon_grid)` → return a 3D numpy
  array shaped `(15, len(lat_grid), len(lon_grid))` — all depths at once,
  batched over the grid. `predict_region_field()` and
  `predict_region_profile_stats()` are already built on top of this, so
  they don't need separate edits.

Keep the function names, arguments, and return shapes identical — `app.py`
does not need any changes.
