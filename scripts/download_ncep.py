"""Download NCEP/NCAR Reanalysis 1 winds for the steering-flow features.

Dataset
-------
NCEP/NCAR Reanalysis 1 (Kalnay et al., 1996), provided by the NOAA Physical
Sciences Laboratory. 6-hourly (00/06/12/18 UTC), 2.5 degree global grid.
Public U.S. government data; no account needed.

What is downloaded
------------------
Zonal (uwnd) and meridional (vwnd) wind at 850, 500 and 200 hPa, for
62.5N-60S (all storms in the project lie between 50S and 52N, and the
steering ring reaches about 7 degrees beyond the storm), for calendar
years 1980-2024. The server cuts out just this slice (OPeNDAP), in 30-day pieces:
whole-year requests get truncated by the server and silently come back as zeros.

Each year is saved to data/raw/ncep/ncep_winds_<year>.nc (both components,
packed to 0.01 m/s and compressed). Years already on disk are skipped, so the
script can simply be re-run after an interruption. Every year is checked for
a complete 6-hourly time axis, missing values and zero-filled (truncated) transfers
before it is saved.

Usage
-----
    conda activate stormtrack
    python scripts/download_ncep.py                      # all years, 1980-2024
    python scripts/download_ncep.py --years 1980 1981    # specific years
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ncep"
URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/pressure/{var}.{year}.nc"
LEVELS = [850, 500, 200]
LAT_N, LAT_S = 62.5, -60.0
YEARS = range(1980, 2025)
RETRIES = 8
PAUSE = 1.5         # seconds between requests: NOAA rate-limits clients (HTTP 429 Too Many Requests)


CHUNK = 120          # time steps per request (30 days); whole-year requests get truncated by the server
MAX_ZERO_SHARE = 1e-3


def fetch_piece(ds: xr.Dataset, var: str, start: int, stop: int) -> xr.DataArray:
    """One slice of time steps. NOAA's server sometimes cuts a transfer short, and the
    netCDF reader then silently fills the rest with zeros instead of raising an error.
    Real winds are almost never exactly 0.0 m/s, so a piece with more than a handful of
    exact zeros is treated as a failed transfer and requested again."""
    for attempt in range(1, RETRIES + 1):
        try:
            time.sleep(PAUSE)
            piece = ds[var].sel(level=LEVELS, lat=slice(LAT_N, LAT_S)).isel(time=slice(start, stop)).load()
            zero_share = float((piece.values == 0).mean())
            if zero_share > MAX_ZERO_SHARE:
                raise IOError(f"{zero_share:.1%} exact zeros (truncated transfer)")
            return piece
        except Exception as e:
            wait = 30 * attempt
            print(f"    {var} steps {start}-{stop}: attempt {attempt} failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{var} steps {start}-{stop}: failed after {RETRIES} attempts")


def open_remote(var: str, year: int) -> xr.Dataset:
    """Open the remote file, waiting and retrying if the server refuses (e.g. rate limiting)."""
    for attempt in range(1, RETRIES + 1):
        try:
            time.sleep(PAUSE)
            return xr.open_dataset(URL.format(var=var, year=year))
        except Exception as e:
            wait = 30 * attempt
            print(f"    open {var} {year}: attempt {attempt} failed ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"open {var} {year}: failed after {RETRIES} attempts")


def fetch(var: str, year: int) -> xr.DataArray:
    """Fetch one variable for one year, in 30-day pieces."""
    with open_remote(var, year) as ds:
        n = ds.sizes["time"]
        return xr.concat([fetch_piece(ds, var, i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)], dim="time")


def check(da: xr.DataArray, year: int) -> None:
    """Make sure the year is complete: every 6-hourly time present, no missing values."""
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 18:00", freq="6h")
    times = pd.DatetimeIndex(da["time"].values)
    if len(times) != len(expected) or not (times == expected).all():
        raise ValueError(f"{da.name} {year}: {len(times)} times, expected {len(expected)}")
    if int(np.isnan(da.values).sum()):
        raise ValueError(f"{da.name} {year}: contains missing values")
    if float((da.values == 0).mean()) > MAX_ZERO_SHARE:
        raise ValueError(f"{da.name} {year}: too many exact zeros (corrupted transfer)")


def download_year(year: int) -> None:
    dest = OUT / f"ncep_winds_{year}.nc"
    if dest.exists():
        print(f"[skip] {year} already downloaded", flush=True)
        return
    t0 = time.time()
    winds = {}
    for var in ("uwnd", "vwnd"):
        da = fetch(var, year)
        check(da, year)
        # Drop the source file's own packing settings (scale/offset, valid range); mixing them
        # with the packing below would corrupt the values when the file is written.
        da.encoding = {}
        da.attrs = {"units": "m/s", "long_name": {"uwnd": "zonal (eastward) wind", "vwnd": "meridional (northward) wind"}[var]}
        winds[var] = da
    ds = xr.Dataset(winds)
    ds.attrs.update(source="NCEP/NCAR Reanalysis 1, NOAA PSL (Kalnay et al., 1996)",
                    subset=f"levels {LEVELS} hPa, lat {LAT_N} to {LAT_S}, 6-hourly")
    enc = {v: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4}
           for v in winds}
    tmp = dest.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.rename(dest)                                # only complete, checked years get the final name
    print(f"[done] {year}: {dest.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="*", default=list(YEARS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = []
    for year in args.years:
        try:
            download_year(year)
        except Exception as e:                       # keep going; re-running the script retries only what's missing
            print(f"[FAIL] {year}: {e}", flush=True)
            failed.append(year)
    if failed:
        print(f"Years still missing (re-run the script to retry): {failed}", flush=True)
    print(f"Finished {len(args.years)} years in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
