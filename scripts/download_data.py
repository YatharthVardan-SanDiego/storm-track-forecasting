"""Download the raw datasets used in this project into data/raw/.

Datasets
--------
1. IBTrACS v04r01, "since1980" CSV (NOAA NCEI). Global tropical cyclone best
   tracks from 1980 onward, ~145 MB. Gahtan et al. (2024); Knapp et al. (2010).
2. HURDAT2 Atlantic best track (NOAA NHC), 1851-2024, ~7 MB.
   Landsea & Franklin (2013). Used only to cross-check Atlantic positions.

Both are public U.S. government data, free for research and educational use.

Usage
-----
    conda activate stormtrack
    python scripts/download_data.py          # skips files that already exist
    python scripts/download_data.py --force  # re-download everything
"""
import argparse
import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "ibtracs.since1980.list.v04r01.csv": (
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
        "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
    ),
    "hurdat2-1851-2024-040425.txt": (
        "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
    ),
}


def download(name: str, url: str, force: bool = False) -> Path:
    """Stream one file to data/raw/<name> and print its size and SHA-256."""
    dest = RAW / name
    if dest.exists() and not force:
        print(f"[skip] {name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                sha.update(chunk)
    tmp.rename(dest)  # only keep complete files
    print(f"[done] {name}: {dest.stat().st_size / 1e6:.1f} MB, sha256={sha.hexdigest()[:16]}...")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        download(name, url, args.force)


if __name__ == "__main__":
    main()
