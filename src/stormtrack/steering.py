"""Steering-flow features from the NCEP/NCAR Reanalysis 1 winds (Kalnay et al., 1996).

A tropical cyclone is carried along by the large-scale wind around it, the
"steering flow". A standard way to measure it is to average the wind in a ring
around the storm, far enough out to leave the storm's own circulation behind
(here 200-800 km from the centre), at several heights in the atmosphere.

For each forecast origin this module averages u (eastward) and v (northward)
wind over that ring at 850, 500 and 200 hPa, at the origin time only. Nothing
after the origin time is used, so the features are available when the forecast
is made. Grid points are weighted by cos(latitude), the area they represent.

Derived features:
  dlm_u, dlm_v            deep-layer mean wind: the average of the three levels (m/s)
  shear_u, shear_v        200 hPa minus 850 hPa wind (m/s)
  steer_dx, steer_dy      how far the deep-layer wind would carry the storm in 24 h (km)
  steer_minus_persist_*   steer_* minus the persistence forecast (km): where the
                          steering flow disagrees with a straight-line extrapolation
"""
import numpy as np
import pandas as pd
import xarray as xr

from .data_io import ROOT
from .geo import haversine_km

NCEP_DIR = ROOT / "data" / "raw" / "ncep"
LEVELS = [850, 500, 200]
RING_KM = (200.0, 800.0)
KM_PER_24H = 86.4                       # 1 m/s sustained for 24 hours = 86.4 km

RING_COLS = [f"{c}{lev}" for lev in LEVELS for c in ("u", "v")]
STEERING_COLS = RING_COLS + ["dlm_u", "dlm_v", "shear_u", "shear_v", "steer_dx", "steer_dy",
                             "steer_minus_persist_dx", "steer_minus_persist_dy"]


def available_years() -> set:
    """Calendar years whose wind file is on disk."""
    return {int(p.stem.split("_")[-1]) for p in NCEP_DIR.glob("ncep_winds_*.nc")}


def _ring_means(year: int, lat: np.ndarray, lon: np.ndarray, times: pd.Series, batch: int = 400) -> np.ndarray:
    """Ring-mean u and v at each level for forecasts that all fall in one calendar year.

    Returns an array (n, 6) ordered like RING_COLS."""
    with xr.open_dataset(NCEP_DIR / f"ncep_winds_{year}.nc") as ds:
        ds = ds.sel(level=LEVELS)
        file_times = pd.DatetimeIndex(ds["time"].values)
        tidx = file_times.get_indexer(pd.DatetimeIndex(times))
        if (tidx < 0).any():
            raise ValueError(f"{year}: {(tidx < 0).sum()} forecast times not found in the wind file")
        u = ds["uwnd"].values.astype(np.float32)          # (time, level, lat, lon)
        v = ds["vwnd"].values.astype(np.float32)
        glat, glon = ds["lat"].values, ds["lon"].values

    LAT, LON = np.meshgrid(glat, glon, indexing="ij")
    LAT, LON = LAT.ravel(), LON.ravel()
    coslat = np.cos(np.radians(LAT)).astype(np.float32)
    n_lev, n_pts = len(LEVELS), LAT.size
    u, v = u.reshape(len(file_times), n_lev, n_pts), v.reshape(len(file_times), n_lev, n_pts)

    out = np.empty((len(lat), 2 * n_lev), dtype=np.float32)
    for s in range(0, len(lat), batch):
        e = min(s + batch, len(lat))
        dist = haversine_km(lat[s:e, None], lon[s:e, None], LAT[None, :], LON[None, :])
        w = ((dist >= RING_KM[0]) & (dist <= RING_KM[1])) * coslat[None, :]
        w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)          # (b, points)
        ub = np.einsum("bp,blp->bl", w, u[tidx[s:e]])
        vb = np.einsum("bp,blp->bl", w, v[tidx[s:e]])
        out[s:e, 0::2], out[s:e, 1::2] = ub, vb
    return out


def steering_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Steering features for every forecast origin in `samples` (which must carry LAT, LON,
    ISO_TIME and the persistence motion dx_m0, dy_m0). Rows whose year has no wind file get NaN."""
    feats = np.full((len(samples), len(RING_COLS)), np.nan, dtype=np.float32)
    years = samples["ISO_TIME"].dt.year.to_numpy()
    have = available_years()
    for year in sorted(set(years) & have):
        m = years == year
        feats[m] = _ring_means(year, samples["LAT"].to_numpy()[m], samples["LON"].to_numpy()[m],
                               samples["ISO_TIME"][m])

    f = pd.DataFrame(feats, columns=RING_COLS, index=samples.index)
    f["dlm_u"] = f[["u850", "u500", "u200"]].mean(axis=1)
    f["dlm_v"] = f[["v850", "v500", "v200"]].mean(axis=1)
    f["shear_u"], f["shear_v"] = f["u200"] - f["u850"], f["v200"] - f["v850"]
    f["steer_dx"], f["steer_dy"] = f["dlm_u"] * KM_PER_24H, f["dlm_v"] * KM_PER_24H
    f["steer_minus_persist_dx"] = f["steer_dx"] - 4 * samples["dx_m0"]
    f["steer_minus_persist_dy"] = f["steer_dy"] - 4 * samples["dy_m0"]
    return f[STEERING_COLS]
