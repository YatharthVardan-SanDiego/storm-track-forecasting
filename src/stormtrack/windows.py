"""Turn cleaned tracks into 24-hour forecast samples, and score forecasts.

A forecast is issued at an *origin* fix i. It may use fixes i-3 .. i (24 hours
of history, i.e. 4 fixes and the 3 six-hour motions between them) and must
predict the position at fix i+4 (24 hours later) in the same continuous segment.

Positions are handled as displacements in km in a local east/north frame
centred on the origin fix (KM_PER_DEG km per degree of latitude, and
KM_PER_DEG * cos(origin latitude) per degree of longitude). Using one frame
for both features and targets means a displacement can be turned back into a
latitude/longitude exactly, and storms crossing the date line need no special care.
"""
import numpy as np
import pandas as pd

from .geo import haversine_km, wrap_lon

KM_PER_DEG = 111.195          # km per degree on a sphere of radius 6371 km
HISTORY = 4                   # fixes of history (24 h)
LEAD = 4                      # steps ahead (24 h)


def build_samples(df: pd.DataFrame, history: int = HISTORY, lead: int = LEAD) -> pd.DataFrame:
    """One row per usable forecast origin, with features, target and metadata.

    `df` is the cleaned track table (one row per 6-hourly fix, column SEG_ID).
    Motion features m0 (most recent), m1, m2 ... are the 6-hour displacements
    between consecutive history fixes, expressed in the origin's km frame.
    """
    d = df.sort_values(["SEG_ID", "ISO_TIME"]).reset_index(drop=True)
    g = d.groupby("SEG_ID")
    i = g.cumcount()
    n = g["LAT"].transform("size")
    origin = (i >= history - 1) & (i <= n - lead - 1)

    lat0 = d["LAT"]
    coslat = np.cos(np.radians(lat0))
    out = pd.DataFrame({
        "SID": d["SID"], "SEG_ID": d["SEG_ID"], "BASIN": d["BASIN"], "SEASON": d["SEASON"],
        "split": d["split"], "ISO_TIME": d["ISO_TIME"], "LAT": lat0, "LON": d["LON"],
    })

    # Motion over each of the history steps, in the origin's km frame
    for k in range(history - 1):
        la_a, lo_a = g["LAT"].shift(k), g["LON"].shift(k)          # later fix of the step
        la_b, lo_b = g["LAT"].shift(k + 1), g["LON"].shift(k + 1)  # earlier fix of the step
        out[f"dx_m{k}"] = wrap_lon(lo_a - lo_b) * KM_PER_DEG * coslat
        out[f"dy_m{k}"] = (la_a - la_b) * KM_PER_DEG

    # Location, season and intensity at the origin
    out["lat"] = lat0
    out["sin_lon"], out["cos_lon"] = np.sin(np.radians(d["LON"])), np.cos(np.radians(d["LON"]))
    doy = d["ISO_TIME"].dt.dayofyear
    out["sin_doy"], out["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["wind_kt"] = d["wind_kt"]
    out["dwind_6h"] = d["wind_kt"] - g["wind_kt"].shift(1)
    out["dwind_18h"] = d["wind_kt"] - g["wind_kt"].shift(history - 1)
    out["dist2land_km"] = d["DIST2LAND"]

    # Target: displacement to the fix `lead` steps ahead, plus the true position for scoring
    la_t, lo_t = g["LAT"].shift(-lead), g["LON"].shift(-lead)
    out["dx_target"] = wrap_lon(lo_t - d["LON"]) * KM_PER_DEG * coslat
    out["dy_target"] = (la_t - lat0) * KM_PER_DEG
    out["lat_true"], out["lon_true"] = la_t, lo_t

    return out[origin.to_numpy()].reset_index(drop=True)


MOTION_COLS = [f"{a}_m{k}" for k in range(HISTORY - 1) for a in ("dx", "dy")]
FEATURE_COLS = MOTION_COLS + ["lat", "sin_lon", "cos_lon", "sin_doy", "cos_doy",
                              "wind_kt", "dwind_6h", "dwind_18h", "dist2land_km"]
TARGET_COLS = ["dx_target", "dy_target"]


def persistence_displacement(samples: pd.DataFrame, lead: int = LEAD) -> np.ndarray:
    """Straight-line extrapolation: repeat the last 6-hour motion `lead` times."""
    return lead * samples[["dx_m0", "dy_m0"]].to_numpy()


def to_latlon(lat0, lon0, dx_km, dy_km):
    """Turn a km displacement in the origin's frame back into latitude/longitude."""
    lat0, lon0 = np.asarray(lat0, float), np.asarray(lon0, float)
    lat = lat0 + np.asarray(dy_km, float) / KM_PER_DEG
    lon = wrap_lon(lon0 + np.asarray(dx_km, float) / (KM_PER_DEG * np.cos(np.radians(lat0))))
    return lat, lon


def track_error_km(samples: pd.DataFrame, pred_disp: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between forecast and actual position, per sample."""
    lat_p, lon_p = to_latlon(samples["LAT"], samples["LON"], pred_disp[:, 0], pred_disp[:, 1])
    return haversine_km(lat_p, lon_p, samples["lat_true"], samples["lon_true"])


STEP_FEATURES = ["dx", "dy", "lat", "wind_kt", "valid"]
STATIC_COLS = ["sin_lon", "cos_lon", "sin_doy", "cos_doy", "dist2land_km"]


def build_sequences(df: pd.DataFrame, samples: pd.DataFrame, n_steps: int = HISTORY - 1) -> np.ndarray:
    """Step-by-step history for each forecast origin in `samples`, for recurrent models.

    Returns an array of shape (n_samples, n_steps, len(STEP_FEATURES)), oldest step
    first. Step k (counting back from the origin) describes the fix k steps before the
    origin: the 6-hour motion that brought the storm there (dx, dy in the origin's km
    frame), its latitude and wind, and a `valid` flag. With the default n_steps = 3 the
    model sees exactly the same 4 fixes as the baselines. Longer histories reach further
    back; where a segment does not go back that far, the step is zero-filled and
    `valid` = 0, so every model is still scored on the same set of origins.
    """
    d = df.sort_values(["SEG_ID", "ISO_TIME"]).reset_index(drop=True)
    g = d.groupby("SEG_ID")
    key = d["SEG_ID"] + "|" + d["ISO_TIME"].astype(str)
    row_of = pd.Series(np.arange(len(d)), index=key)
    rows = row_of.loc[samples["SEG_ID"] + "|" + samples["ISO_TIME"].astype(str)].to_numpy()

    coslat0 = np.cos(np.radians(d["LAT"].to_numpy()))[rows]
    seq = np.zeros((len(rows), n_steps, len(STEP_FEATURES)), dtype=np.float32)
    for k in range(n_steps):
        la_a, lo_a = g["LAT"].shift(k).to_numpy()[rows], g["LON"].shift(k).to_numpy()[rows]
        la_b, lo_b = g["LAT"].shift(k + 1).to_numpy()[rows], g["LON"].shift(k + 1).to_numpy()[rows]
        wind = g["wind_kt"].shift(k).to_numpy()[rows]
        valid = ~np.isnan(la_b)
        feats = np.stack([wrap_lon(lo_a - lo_b) * KM_PER_DEG * coslat0, (la_a - la_b) * KM_PER_DEG,
                          la_a, wind, valid.astype(float)], axis=1)
        seq[:, n_steps - 1 - k, :] = np.where(valid[:, None], np.nan_to_num(feats), 0.0)
    return seq
