"""Geographic helpers used across the project.

All distances are great-circle (haversine) distances on a spherical Earth,
which is the standard way tropical cyclone track errors are reported
(e.g., the NHC verification reports track error in nautical miles / km).
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def wrap_lon(lon):
    """Wrap longitudes to the range [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in decimal degrees.

    Works element-wise on scalars, NumPy arrays, or pandas Series.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing (0 = north, 90 = east) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
