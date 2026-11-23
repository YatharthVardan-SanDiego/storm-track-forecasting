"""Readers for the two raw datasets (IBTrACS CSV and HURDAT2 text)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

IBTRACS_CSV = RAW / "ibtracs.since1980.list.v04r01.csv"
HURDAT2_TXT = RAW / "hurdat2-1851-2024-040425.txt"

# Only the IBTrACS columns this project uses (the file has 174).
IBTRACS_COLUMNS = [
    "SID", "SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "NATURE",
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "WMO_AGENCY", "TRACK_TYPE",
    "DIST2LAND", "LANDFALL", "IFLAG", "USA_AGENCY", "USA_ATCF_ID",
    "USA_WIND", "USA_PRES", "USA_SSHS", "STORM_SPEED", "STORM_DIR",
]


def load_ibtracs(path=IBTRACS_CSV, columns=IBTRACS_COLUMNS) -> pd.DataFrame:
    """Load the IBTrACS CSV.

    The file has a second header row with units, which is skipped. Missing
    values are stored as a single space, so blanks are read as NaN. The
    basin code "NA" (North Atlantic) must NOT be read as missing, so pandas'
    default NA strings are switched off.
    """
    df = pd.read_csv(
        path, skiprows=[1], usecols=columns,
        keep_default_na=False, na_values=["", " "], low_memory=False,
    )
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    return df


def load_hurdat2(path=HURDAT2_TXT) -> pd.DataFrame:
    """Parse HURDAT2 into one row per best-track fix.

    HURDAT2 alternates a header line per storm (ATCF id, name, number of
    lines) with that many data lines. Positions are given like "28.0N" and
    "94.8W" and are converted to signed decimal degrees.
    """
    rows = []
    with open(path) as f:
        lines = iter(f)
        for header in lines:
            atcf_id, name, n = [p.strip() for p in header.split(",")[:3]]
            for _ in range(int(n)):
                p = [v.strip() for v in next(lines).split(",")]
                lat = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
                lon = float(p[5][:-1]) * (1 if p[5][-1] == "E" else -1)
                rows.append({
                    "ATCF_ID": atcf_id, "NAME": name,
                    "ISO_TIME": pd.Timestamp(f"{p[0]} {p[1][:2]}:{p[1][2:]}"),
                    "RECORD_ID": p[2], "STATUS": p[3], "LAT": lat, "LON": lon,
                    "WIND": int(p[6]), "PRES": int(p[7]),
                })
    df = pd.DataFrame(rows)
    df.loc[df["PRES"] < 0, "PRES"] = pd.NA  # -999 = missing
    return df
