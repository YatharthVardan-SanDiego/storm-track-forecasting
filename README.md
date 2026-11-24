# Learning the Path of the Storm
### 24-Hour Tropical Cyclone Track Forecasting with Deep Learning

AAI-590 Capstone Project · Master of Science in Applied Artificial Intelligence, University of San Diego
Author: Yatharth Vardan

This project asks how well a storm's next-day position can be predicted from its own recent track history.
It cleans the global IBTrACS best-track archive, then compares simple baselines (persistence, XGBoost) against
recurrent neural networks (LSTM, GRU) on held-out recent seasons, measuring great-circle track error in km
against the National Hurricane Center's official 24-hour error as a benchmark.

## Project status

| Stage (rubric section) | Notebook | Status |
|---|---|---|
| Data Cleaning | `notebooks/01_data_cleaning.ipynb` | Done |
| EDA | `notebooks/02_eda.ipynb` | Done |
| Model / Pipeline Design & Building | `notebooks/03_...` | Planned |
| Model Training | | Planned |
| Model Optimization | | Planned |
| Model / Pipeline Analysis & Discussion | | Planned |

## Data

| Dataset | Source | Use |
|---|---|---|
| IBTrACS v04r01, "since 1980" CSV (145 MB) | NOAA NCEI (Gahtan et al., 2024; Knapp et al., 2010) | Main dataset: global storm positions, winds and pressures every 6 hours |
| HURDAT2 Atlantic best track, 1851–2024 (7 MB) | NOAA NHC (Landsea & Franklin, 2013) | Cross-check of Atlantic positions |

Both are public U.S. government datasets. They are **not** stored in this repository; download them with the
script below.

## How to run

```bash
conda env create -f environment.yml
conda activate stormtrack
python scripts/download_data.py
jupyter notebook   # then run notebooks/01_data_cleaning.ipynb, then 02_eda.ipynb
```

## Files

| Path | Description |
|---|---|
| `README.md` | This file: project overview, how to run, and a description of every file. |
| `environment.yml` | Conda environment (Python 3.12, pandas, NumPy, Matplotlib, Jupyter). Modeling libraries are added when the modeling notebooks are added. |
| `.gitignore` | Keeps raw/processed data, checkpoints and caches out of git. |
| `scripts/download_data.py` | Downloads IBTrACS and HURDAT2 into `data/raw/`, skipping files already present and printing size and checksum. |
| `src/stormtrack/__init__.py` | Marks `stormtrack` as a Python package of shared helpers. |
| `src/stormtrack/data_io.py` | Loaders: `load_ibtracs()` reads the IBTrACS CSV (skips the units row, keeps basin code "NA" as text); `load_hurdat2()` parses the HURDAT2 text format into one row per fix; `load_clean()` reads the cleaned table from notebook 01 (also keeping "NA" as text). |
| `src/stormtrack/geo.py` | Geographic helpers: haversine great-circle distance (km), compass bearing, and longitude wrapping to [-180, 180). |
| `notebooks/01_data_cleaning.ipynb` | **Data Cleaning.** Keeps finalized main tracks through 2024, 6-hourly synoptic fixes, and tropical/subtropical stages; builds one 1-minute wind column and one pressure column; splits storms into continuous segments of at least 42 h; adds motion features; cross-checks Atlantic positions with HURDAT2; assigns season-based train/val/test splits; saves the clean table and a cleaning log. |
| `notebooks/02_eda.ipynb` | **Exploratory Data Analysis.** Data volume by basin, season and split; track map; speed and heading and how they change with latitude; autocorrelation of motion; intensity and missing pressure; the 24-hour displacement target and the error of two naive forecasts (no motion, persistence); comparison of the splits. Ends with a table of modeling decisions. |
| `reports/figures/fig01_cleaning_funnel.png` | Figure: fixes remaining after each cleaning step (for the report's Data Summary). |
| `reports/figures/fig02_storms_per_season.png` | Figure: storms per season by basin, with split boundaries. |
| `reports/figures/fig03_track_map.png` | Figure: all clean tracks, colored by basin. |
| `reports/figures/fig04_speed_heading.png` | Figure: translation speed histogram and heading roses for each hemisphere. |
| `reports/figures/fig05_motion_by_latitude.png` | Figure: eastward/poleward share and median speed by distance from the equator (recurvature). |
| `reports/figures/fig06_motion_autocorrelation.png` | Figure: how persistent storm motion is over 6–72 h lags. |
| `reports/figures/fig07_intensity.png` | Figure: wind distribution, wind vs pressure, missing pressure by basin. |
| `reports/figures/fig08_target_and_naive_errors.png` | Figure: 24-h displacement target and errors of the no-motion and persistence forecasts. |
| `reports/figures/fig09_persistence_error_drivers.png` | Figure: persistence error by latitude and by turning rate. |
| `reports/figures/fig10_split_comparison.png` | Figure: basin mix, speed and wind across train/val/test. |
| `data/raw/` | Raw downloads (not tracked). |
| `data/processed/` | Notebook outputs, not tracked: `tracks_clean.csv.gz` (105,747 clean fixes) `cleaning_log.csv` (rows kept at each step), and `eda_baselines.csv` (naive-forecast errors by split, from notebook 02). |
| `reports/notebook_pdfs/` | PDF exports of each notebook for the final submission (added at the end). |

## Cleaning results (summary)

312,480 raw rows → **105,747 clean 6-hourly fixes** in 4,162 continuous segments from 4,021 storms (1980–2024).

| Split | Seasons | Storms | Fixes |
|---|---|---|---|
| Train | 1980–2016 | 3,302 | 88,941 |
| Validation | 2017–2019 | 275 | 6,757 |
| Test | 2020–2024 | 444 | 10,049 |

## EDA highlights

- 76,613 usable 24-hour forecast samples (4 fixes of history + the fix 24 h later): 65,169 train, 4,699 validation, 6,745 test.
- Storm motion is highly persistent (lag-1 correlation 0.93–0.95), but direction and speed change sharply with latitude: past 30° most storms turn east and speed up (recurvature).
- Naive 24-hour forecasts on the test seasons: no motion ≈ 399 km mean error; persistence ≈ 168 km (≈ 179 km North Atlantic only). NHC's official Atlantic error for 2020–2024 is about 63 km.
- Persistence error roughly doubles beyond 30° latitude and rises when the storm is already turning: the cases a learned model should improve on.

## References

Gahtan, J., Knapp, K. R., Schreck, C. J., Diamond, H. J., Kossin, J. P., & Kruk, M. C. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01*. NOAA National Centers for Environmental Information. https://doi.org/10.25921/82ty-9e16

Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., & Neumann, C. J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society, 91*(3), 363–376.

Landsea, C. W., & Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new database format. *Monthly Weather Review, 141*(10), 3576–3592.

National Hurricane Center. (2025). *Official Atlantic basin annual average track errors, 1989–present* [Data set]. National Oceanic and Atmospheric Administration. https://www.nhc.noaa.gov/verification/pdfs/1989-present_OFCL_ATL_annual_trk_errors.pdf

## AI-tool disclosure

Claude (Anthropic) was used to help with documentation in this repository. All code was supervised , reviewed, run, and checked by the author.
