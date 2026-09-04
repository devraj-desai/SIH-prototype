"""
model_interface.py
-------------------
This module is the ONLY place that needs to change once the real
Satellite-Embedding Deep Learning model is ready.

Two functions are called by the Streamlit app (app.py):

    1. predict_temperature_profile(...)  -> single-point depth profile
    2. predict_region_field(...)         -> whole-region 2D field at one depth

Keep the function names, arguments, and return formats IDENTICAL when you
plug in the real model — that is what lets the rest of the app work
unmodified. Everything inside the functions below is mock / placeholder
logic and should be deleted and replaced with real inference code
(load model weights once at import time, build the input tensor from the
arguments, run model.predict / model.forward, and return the result in
the same shape).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixed constants describing the problem statement's domain & output levels
# ---------------------------------------------------------------------------

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

REGION_BOUNDS = {
    "lat_min": 5.0,
    "lat_max": 30.0,
    "lon_min": 45.0,
    "lon_max": 105.0,
}

GRID_RESOLUTION_DEG = 0.25  # matches the problem statement's target resolution


# ---------------------------------------------------------------------------
# 1. POINT-WISE PROFILE RECONSTRUCTION
# ---------------------------------------------------------------------------

def predict_temperature_profile(lat, lon, sst, sss, ssh, uo, vo, date):
    """
    Reconstruct the subsurface temperature profile at a single lat/lon point.

    Parameters
    ----------
    lat, lon : float
        Location within REGION_BOUNDS (degrees).
    sst : float
        Sea Surface Temperature (deg C).
    sss : float
        Sea Surface Salinity (PSU).
    ssh : float
        Sea Surface Height / Sea Level Anomaly (m).
    uo, vo : float
        Surface current components (m/s).
    date : datetime.date
        Observation date.

    Returns
    -------
    dict
        {
          "depths":      list[float]  -- STANDARD_DEPTHS,
          "temperature": list[float]  -- reconstructed temperature (deg C)
                                          at each depth level
        }

    >>> REPLACE THE BODY BELOW WITH A CALL TO THE TRAINED MODEL <<<
    e.g.
        embedding = encoder(sst, sss, ssh, uo, vo, lat, lon, date)
        temperature = decoder(embedding, STANDARD_DEPTHS)
        return {"depths": STANDARD_DEPTHS, "temperature": temperature.tolist()}
    """
    depths = np.array(STANDARD_DEPTHS, dtype=float)

    # ---------------------- MOCK LOGIC (delete when real model is wired in) ----
    decay = np.exp(-depths / 400.0)
    seasonal = 0.5 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.0)
    spatial = 0.3 * np.sin(np.radians(lat)) - 0.2 * np.cos(np.radians(lon))
    current_effect = 0.4 * (uo - vo)
    salinity_effect = -0.05 * (sss - 35.0)
    ssh_effect = 0.8 * ssh

    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 1000)))
    noise = rng.normal(0, 0.05, size=depths.shape)

    temperature = (
        sst * decay
        + seasonal
        + spatial
        + current_effect * decay
        + salinity_effect * decay
        + ssh_effect * decay
        + noise
    )
    temperature = np.clip(temperature, -2.0, sst + 2.0)
    # -----------------------------------------------------------------------

    return {"depths": depths.tolist(), "temperature": temperature.tolist()}


# ---------------------------------------------------------------------------
# 2. WHOLE-REGION FIELD RECONSTRUCTION (used when no point is selected)
# ---------------------------------------------------------------------------

def predict_region_volume(date, lat_grid, lon_grid):
    """
    Reconstruct the FULL 3D temperature volume (all standard depths at
    once) over the region for one date. This is the batched equivalent of
    predict_temperature_profile() and is what both the map overlay and the
    region-selection statistics are built from.

    Parameters
    ----------
    date : datetime.date
    lat_grid, lon_grid : 1-D np.ndarray
        Latitude / longitude coordinates of the output grid.

    Returns
    -------
    np.ndarray
        3D array of shape (len(STANDARD_DEPTHS), len(lat_grid), len(lon_grid))
        with reconstructed temperature (deg C).

    >>> REPLACE THE BODY BELOW WITH A CALL TO THE TRAINED MODEL <<<
    e.g. batch the surface fields (SST/SSS/SSH/U/V rasters) for `date`,
    run the model once over the whole grid for all depths, and reshape to
    (n_depth, nlat, nlon).
    """
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # ---------------------- MOCK LOGIC (delete when real model is wired in) ----
    seasonal = 0.5 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.0)
    base = 29.0 - 0.18 * (lat_mesh - REGION_BOUNDS["lat_min"])
    eddies = 1.6 * np.sin(lat_mesh / 2.0) * np.cos(lon_mesh / 3.0 + date.toordinal() / 50.0)
    surface_field = base + eddies + seasonal

    volume = np.stack(
        [surface_field * np.exp(-d / 400.0) for d in STANDARD_DEPTHS],
        axis=0,
    )
    # -----------------------------------------------------------------------

    return volume


def predict_region_field(depth, date, lat_grid, lon_grid):
    """
    Reconstruct a 2D temperature field over the full region at one depth
    level and date. Used to draw the basin-wide map overlay.

    Thin convenience wrapper around predict_region_volume() that slices out
    the nearest standard depth. Kept as a separate function so the app can
    request a single depth layer without needing the whole volume.

    Returns
    -------
    np.ndarray
        2D array of shape (len(lat_grid), len(lon_grid)) with reconstructed
        temperature (deg C).
    """
    volume = predict_region_volume(date, lat_grid, lon_grid)
    depth_idx = int(np.argmin(np.abs(np.array(STANDARD_DEPTHS) - depth)))
    return volume[depth_idx]


def predict_region_profile_stats(lat_min, lat_max, lon_min, lon_max, date, resolution=GRID_RESOLUTION_DEG):
    """
    Aggregate statistics (mean/min/max/std) of the reconstructed temperature
    profile over a rectangular sub-region, at every standard depth. Used by
    the "draw a region" selection mode.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : float
        Bounds of the selected rectangle (already clipped to REGION_BOUNDS).
    date : datetime.date
    resolution : float
        Grid spacing (degrees) used to sample the sub-region.

    Returns
    -------
    dict with keys:
        "depths" : list[float]
        "mean", "min", "max", "std" : list[float]  -- one value per depth
        "n_points" : int  -- number of grid cells sampled
    """
    lat_sub = np.arange(lat_min, lat_max + 1e-6, resolution)
    lon_sub = np.arange(lon_min, lon_max + 1e-6, resolution)
    if len(lat_sub) == 0:
        lat_sub = np.array([lat_min])
    if len(lon_sub) == 0:
        lon_sub = np.array([lon_min])

    volume = predict_region_volume(date, lat_sub, lon_sub)  # (n_depth, nlat, nlon)
    flat = volume.reshape(volume.shape[0], -1)

    return {
        "depths": list(STANDARD_DEPTHS),
        "mean": flat.mean(axis=1).tolist(),
        "min": flat.min(axis=1).tolist(),
        "max": flat.max(axis=1).tolist(),
        "std": flat.std(axis=1).tolist(),
        "n_points": int(flat.shape[1]),
    }
