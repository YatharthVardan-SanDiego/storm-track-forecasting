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
| Model / Pipeline Design & Building | `notebooks/03_baselines.ipynb` (shared pipeline + baselines) | Done |
| Model Training | `notebooks/03_baselines.ipynb` (baselines), `notebooks/04_lstm_gru.ipynb` (LSTM/GRU) | Done |
| Model Optimization | `notebooks/05_tuning.ipynb` | Done |
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
jupyter notebook   # then run notebooks 01 to 05 in order
```

## Files

| Path | Description |
|---|---|
| `README.md` | This file: project overview, how to run, and a description of every file. |
| `environment.yml` | Conda environment (Python 3.12, pandas, NumPy, Matplotlib, Jupyter, scikit-learn, XGBoost, PyTorch). |
| `.gitignore` | Keeps raw/processed data, checkpoints and caches out of git. |
| `scripts/download_data.py` | Downloads IBTrACS and HURDAT2 into `data/raw/`, skipping files already present and printing size and checksum. |
| `src/stormtrack/__init__.py` | Marks `stormtrack` as a Python package of shared helpers. |
| `src/stormtrack/data_io.py` | Loaders: `load_ibtracs()` reads the IBTrACS CSV (skips the units row, keeps basin code "NA" as text); `load_hurdat2()` parses the HURDAT2 text format into one row per fix; `load_clean()` reads the cleaned table from notebook 01 (also keeping "NA" as text). |
| `src/stormtrack/windows.py` | Forecast pipeline shared by all models: `build_samples()` turns tracks into 24-h forecast samples (4 fixes of history → position 24 h later) with motion, location, season, intensity and land features; `persistence_displacement()`; `to_latlon()` and `track_error_km()` turn predicted displacements into positions and great-circle errors; `build_sequences()` lays out each sample's history step by step for the recurrent models. |
| `src/stormtrack/rnn.py` | Recurrent forecaster (`RNNForecaster`: LSTM or GRU encoder + static features + MLP head predicting a correction to persistence), the distance loss (mean km error), the training loop with early stopping (optimizing distance or squared error, always early-stopped on km error), and batched prediction. |
| `src/stormtrack/geo.py` | Geographic helpers: haversine great-circle distance (km), compass bearing, and longitude wrapping to [-180, 180). |
| `notebooks/01_data_cleaning.ipynb` | **Data Cleaning.** Keeps finalized main tracks through 2024, 6-hourly synoptic fixes, and tropical/subtropical stages; builds one 1-minute wind column and one pressure column; splits storms into continuous segments of at least 42 h; adds motion features; cross-checks Atlantic positions with HURDAT2; assigns season-based train/val/test splits; saves the clean table and a cleaning log. |
| `notebooks/02_eda.ipynb` | **Exploratory Data Analysis.** Data volume by basin, season and split; track map; speed and heading and how they change with latitude; autocorrelation of motion; intensity and missing pressure; the 24-hour displacement target and the error of two naive forecasts (no motion, persistence); comparison of the splits. Ends with a table of modeling decisions. |
| `notebooks/03_baselines.ipynb` | **Baseline models.** Builds forecast samples; trains and compares no-motion, persistence, CLIPER-style linear regression and XGBoost on the validation seasons; tests direct vs residual targets; storm-level paired bootstrap; error by latitude; feature importance; an example storm (Jose, 2017). Saves validation predictions and the XGBoost models. |
| `notebooks/04_lstm_gru.ipynb` | **LSTM and GRU.** Builds step-by-step sequences, trains an LSTM and a GRU (PyTorch, Apple GPU) on a distance loss with early stopping, checks stability over 3 seeds, compares 3-seed ensembles with the baselines (bootstrap, latitude, Hurricane Jose). Note: XGBoost and PyTorch are never loaded in the same session (their OpenMP libraries conflict on macOS). |
| `notebooks/05_tuning.ipynb` | **Model optimization.** One-at-a-time search on the validation seasons: distance vs squared-error loss, 24/48/72-hour history (padded with a `valid` flag so all models forecast the same origins), hidden size and depth; ties within 0.3 km go to the simpler option. Trains the tuned GRU and LSTM with 3 seeds and compares them with all earlier models. |
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
| `reports/figures/fig11_baseline_val_errors.png` | Figure: mean validation error of the four baselines, all basins and North Atlantic, vs NHC. |
| `reports/figures/fig12_error_by_latitude_baselines.png` | Figure: validation error by latitude band for persistence, linear regression and XGBoost. |
| `reports/figures/fig13_xgb_feature_importance.png` | Figure: XGBoost feature importance (share of total gain). |
| `reports/figures/fig14_example_storm_forecasts.png` | Figure: persistence and XGBoost 24-h forecasts along Hurricane Jose (2017). |
| `reports/figures/fig15_rnn_learning_curves.png` | Figure: LSTM and GRU training/validation loss per epoch. |
| `reports/figures/fig16_error_by_latitude_rnn.png` | Figure: validation error by latitude band, baselines and recurrent networks. |
| `reports/figures/fig17_jose_xgb_vs_lstm.png` | Figure: XGBoost and LSTM 24-h forecasts along Hurricane Jose (2017). |
| `reports/figures/fig18_tuning_results.png` | Figure: validation error for every tuning run, by experiment. |
| `reports/figures/fig19_tuned_vs_all.png` | Figure: validation error of all models before and after tuning. |
| `models/` | Saved models, not tracked: `xgb_dx.json`, `xgb_dy.json`, `xgb_config.json` (notebook 03); `lstm_seed*.pt`, `gru_seed*.pt`, `rnn_config.json` (notebook 04); `tuned_gru_seed*.pt`, `tuned_lstm_seed*.pt`, `tuned_config.json` (notebook 05). |
| `data/raw/` | Raw downloads (not tracked). |
| `data/processed/` | Notebook outputs, not tracked: `tracks_clean.csv.gz` (105,747 clean fixes), `cleaning_log.csv` (rows kept at each step), `eda_baselines.csv` (naive-forecast errors by split, from notebook 02), `preds_val_baselines.csv` (per-forecast validation errors and positions of each baseline, from notebook 03), `preds_val_rnn.csv` (per-forecast validation errors of the LSTM and GRU ensembles, from notebook 04), `preds_val_tuned.csv` and `tuning_log.csv` (tuned ensembles' errors and every tuning run, from notebook 05). |
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

## Results so far (validation seasons 2017–2019)

| Model | Mean 24-h error | North Atlantic | Skill vs persistence |
|---|---|---|---|
| No motion | 391 km | 462 km | −133% |
| Persistence | 168 km | 183 km | 0% |
| Linear regression (CLIPER-style) | 158 km | 166 km | 5.8% |
| XGBoost (residual target) | 148.5 km | 164 km | 11.6% |
| LSTM (3-seed ensemble) | 146.0 km | 157 km | 13.0% |
| GRU (3-seed ensemble) | 145.2 km | 156 km | 13.5% |
| LSTM tuned (48-h history, 128 units) | 144.3 km | 155 km | 14.1% |
| **GRU tuned (48-h history, 128 units)** | **144.0 km** | **156 km** | **14.3%** |

The test seasons (2020–2024) are held out for the final evaluation. NHC's official Atlantic 24-hour error for 2020–2024 is about 63 km.

## References

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 1724–1734). https://doi.org/10.3115/v1/D14-1179

Gahtan, J., Knapp, K. R., Schreck, C. J., Diamond, H. J., Kossin, J. P., & Kruk, M. C. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01*. NOAA National Centers for Environmental Information. https://doi.org/10.25921/82ty-9e16

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., & Neumann, C. J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society, 91*(3), 363–376.

Landsea, C. W., & Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new database format. *Monthly Weather Review, 141*(10), 3576–3592.

National Hurricane Center. (2025). *Official Atlantic basin annual average track errors, 1989–present* [Data set]. National Oceanic and Atmospheric Administration. https://www.nhc.noaa.gov/verification/pdfs/1989-present_OFCL_ATL_annual_trk_errors.pdf

Neumann, C. J. (1972). *An alternate to the HURRAN tropical cyclone forecast system* (NOAA Technical Memorandum NWS SR-62). National Weather Service.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

## AI-tool disclosure

Claude (Anthropic) was used to help with documentation in this repository. All code was supervised , reviewed, run, and checked by the author.
