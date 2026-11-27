# Learning the Path of the Storm
### 24-Hour Tropical Cyclone Track Forecasting with Deep Learning

AAI-590 Capstone Project · Master of Science in Applied Artificial Intelligence, University of San Diego
Author: Yatharth Vardan

This project asks how well a storm's next-day position can be predicted from its own recent track history.
It cleans the global IBTrACS best-track archive, then compares simple baselines (persistence, XGBoost) against
recurrent neural networks (LSTM, GRU) on held-out recent seasons, measuring great-circle track error in km
against the National Hurricane Center's official 24-hour error as a benchmark.

The work is organised as two studies:

- **Study A (notebooks 01–06, final evaluation in 08a–08b):** track-only forecasting on the full 1980–2024 record, with a locked 2020–2024 test set.
- **Study B (notebooks 07a–07b):** a controlled experiment that adds steering winds from a second dataset, the
  NCEP/NCAR Reanalysis 1, on its own chronological split of the years where both datasets are available.

## Project status

| Stage (rubric section) | Notebook | Status |
|---|---|---|
| Data Cleaning | `notebooks/01_data_cleaning.ipynb` | Done |
| EDA | `notebooks/02_eda.ipynb` | Done |
| Model / Pipeline Design & Building | `notebooks/03_baselines.ipynb` (shared pipeline + baselines) | Done |
| Model Training | `notebooks/03_baselines.ipynb` (baselines), `notebooks/04_lstm_gru.ipynb` (LSTM/GRU) | Done |
| Model Optimization | `notebooks/05_tuning.ipynb` (round 1), `notebooks/06_tuning_bayesian.ipynb` (round 2) | Done |
| Data Sources, Feature Engineering, Training, Evaluation (Study B: steering winds) | `notebooks/07a_steering_winds_xgboost.ipynb`, `notebooks/07b_steering_winds_gru.ipynb` | Done |
| Model / Pipeline Analysis & Discussion (Study A final test) | `notebooks/08a_final_test_baselines.ipynb`, `notebooks/08b_final_test_analysis.ipynb` | Done |

## Data

| Dataset | Source | Use |
|---|---|---|
| IBTrACS v04r01, "since 1980" CSV (145 MB) | NOAA NCEI (Gahtan et al., 2024; Knapp et al., 2010) | Main dataset: global storm positions, winds and pressures every 6 hours |
| HURDAT2 Atlantic best track, 1851–2024 (7 MB) | NOAA NHC (Landsea & Franklin, 2013) | Cross-check of Atlantic positions |
| NCEP/NCAR Reanalysis 1 winds, 850/500/200 hPa, 6-hourly, 2.5° (about 76 MB per year after subsetting) | NOAA PSL (Kalnay et al., 1996) | Study B: steering-flow features around each storm |

All are public U.S. government datasets. They are **not** stored in this repository; download them with the
scripts below. The NCEP download covers 1980–2012 except 2001 and 2009: NOAA's server began rate-limiting
requests (HTTP 429) partway through, so Study B uses the 31 years that arrived.

## How to run

```bash
conda env create -f environment.yml
conda activate stormtrack
python scripts/download_data.py
python scripts/download_ncep.py   # NCEP winds for Study B; re-run to resume after an interruption
jupyter notebook                  # then run notebooks 01 to 08b in order
```

## Files

| Path | Description |
|---|---|
| `README.md` | This file: project overview, how to run, and a description of every file. |
| `environment.yml` | Conda environment (Python 3.12, pandas, NumPy, Matplotlib, Jupyter, scikit-learn, XGBoost, PyTorch, Optuna, xarray, netCDF4). |
| `.gitignore` | Keeps raw/processed data, checkpoints and caches out of git. |
| `scripts/download_ncep.py` | Downloads NCEP/NCAR Reanalysis 1 u/v winds at 850, 500 and 200 hPa (62.5°N–60°S) one year at a time via OPeNDAP, in 30-day pieces with pauses and retries; checks every year for a complete 6-hourly time axis, missing values and truncated (zero-filled) transfers; saves `data/raw/ncep/ncep_winds_<year>.nc`; skips years already on disk. |
| `scripts/export_notebook_pdfs.py` | Exports every notebook to PDF from its saved outputs (nothing is re-run) via headless Google Chrome, fitting wide tables and long outputs to the page width; writes `reports/notebook_pdfs/<notebook name>.pdf`. |
| `scripts/make_report_figures.py` | Redraws the main results at one-column size for the final report (track map, persistence by latitude, network architecture, final test results, error by latitude, Katrina steering flow, Study B results) from the files the notebooks saved; writes `reports/figures/report/`. |
| `reports/figures/report/` | The seven column-sized figures used in the final report (`r1_`–`r7_`), produced by `scripts/make_report_figures.py`. |
| `scripts/download_data.py` | Downloads IBTrACS and HURDAT2 into `data/raw/`, skipping files already present and printing size and checksum. |
| `src/stormtrack/__init__.py` | Marks `stormtrack` as a Python package of shared helpers. |
| `src/stormtrack/data_io.py` | Loaders: `load_ibtracs()` reads the IBTrACS CSV (skips the units row, keeps basin code "NA" as text); `load_hurdat2()` parses the HURDAT2 text format into one row per fix; `load_clean()` reads the cleaned table from notebook 01 (also keeping "NA" as text). |
| `src/stormtrack/windows.py` | Forecast pipeline shared by all models: `build_samples()` turns tracks into 24-h forecast samples (4 fixes of history → position 24 h later) with motion, location, season, intensity and land features; `persistence_displacement()`; `to_latlon()` and `track_error_km()` turn predicted displacements into positions and great-circle errors; `build_sequences()` lays out each sample's history step by step for the recurrent models. |
| `src/stormtrack/rnn.py` | Recurrent forecaster (`RNNForecaster`: LSTM or GRU encoder + static features + MLP head predicting a correction to persistence), the distance loss (mean km error), the training loop with early stopping (optimizing distance or squared error, always early-stopped on km error; an `on_epoch` hook lets a hyperparameter search prune runs), and batched prediction. |
| `src/stormtrack/steering.py` | Steering-flow features: area-weighted mean u/v wind in a 200–800 km ring around each storm at 850, 500 and 200 hPa, at the forecast time only; deep-layer mean wind, vertical shear, 24-h steering displacement and its difference from persistence. |
| `src/stormtrack/geo.py` | Geographic helpers: haversine great-circle distance (km), compass bearing, and longitude wrapping to [-180, 180). |
| `notebooks/01_data_cleaning.ipynb` | **Data Cleaning.** Keeps finalized main tracks through 2024, 6-hourly synoptic fixes, and tropical/subtropical stages; builds one 1-minute wind column and one pressure column; splits storms into continuous segments of at least 42 h; adds motion features; cross-checks Atlantic positions with HURDAT2; assigns season-based train/val/test splits; saves the clean table and a cleaning log. |
| `notebooks/02_eda.ipynb` | **Exploratory Data Analysis.** Data volume by basin, season and split; track map; speed and heading and how they change with latitude; autocorrelation of motion; intensity and missing pressure; the 24-hour displacement target and the error of two naive forecasts (no motion, persistence); comparison of the splits. Ends with a table of modeling decisions. |
| `notebooks/03_baselines.ipynb` | **Baseline models.** Builds forecast samples; trains and compares no-motion, persistence, CLIPER-style linear regression and XGBoost on the validation seasons; tests direct vs residual targets; storm-level paired bootstrap; error by latitude; feature importance; an example storm (Jose, 2017). Saves validation predictions and the XGBoost models. |
| `notebooks/04_lstm_gru.ipynb` | **LSTM and GRU.** Builds step-by-step sequences, trains an LSTM and a GRU (PyTorch, Apple GPU) on a distance loss with early stopping, checks stability over 3 seeds, compares 3-seed ensembles with the baselines (bootstrap, latitude, Hurricane Jose). Note: XGBoost and PyTorch are never loaded in the same session (their OpenMP libraries conflict on macOS). |
| `notebooks/05_tuning.ipynb` | **Model optimization.** One-at-a-time search on the validation seasons: distance vs squared-error loss, 24/48/72-hour history (padded with a `valid` flag so all models forecast the same origins), hidden size and depth; ties within 0.3 km go to the simpler option. Trains the tuned GRU and LSTM with 3 seeds and compares them with all earlier models. |
| `notebooks/06_tuning_bayesian.ipynb` | **Model optimization, round 2.** Bayesian search with Optuna (TPE sampler + median pruning, 40 trials) over cell type, history, width, depth, dropout, learning rate, weight decay and batch size, jointly. Trials train on 1980–2013 and are scored on 2014–2016, so the validation seasons stay untouched; fANOVA hyperparameter importance; the best configuration is trained with 3 seeds and compared with round 1. |
| `notebooks/07a_steering_winds_xgboost.ipynb` | **Study B, part 1.** Introduces the NCEP winds; builds the Study B split (train 1980–2004, validation 2005–2008, test 2009–2013); computes steering features and checks them physically (Katrina, 2005); **EDA of the wind data** (coverage by season and basin, balance across the Study B splits, correlations among wind features and with persistence error, steering wind vs storm motion by latitude, beta drift); compares persistence, steering-only, ridge and XGBoost with and without winds on validation and test, with storm-level bootstrap; feature importance. |
| `notebooks/07b_steering_winds_gru.ipynb` | **Study B, part 2.** Trains the round 2 GRU (unchanged settings, 3 seeds) with and without winds on the Study B split; compares all models on validation and test with bootstrap intervals; error by latitude; an example storm (Nadine, 2012). |
| `notebooks/08a_final_test_baselines.ipynb` | **Final evaluation, part 1.** Reloads the baselines (XGBoost from disk, ridge refit), proves they reproduce notebook 03's validation scores exactly, then scores no motion, persistence, linear regression and XGBoost on the locked 2020–2024 test seasons. |
| `notebooks/08b_final_test_analysis.ipynb` | **Final evaluation, part 2.** Reloads all five neural ensembles (notebooks 04–06) and checks them against their saved validation errors; final test table with bootstrap intervals; validation-vs-test consistency; comparison with NHC's official Atlantic error by season; errors by basin, latitude, intensity and season; error distribution; profile of the worst 1% of forecasts; an example storm (Franklin, 2023). |
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
| `reports/figures/fig20_optuna_history.png` | Figure: Optuna search history (completed and pruned trials, best so far). |
| `reports/figures/fig21_optuna_importance.png` | Figure: hyperparameter importance (fANOVA). |
| `reports/figures/fig22_round2_by_latitude.png` | Figure: validation error by latitude, XGBoost vs round 1 vs round 2 GRU. |
| `reports/figures/fig23b_wind_coverage.png` | Figure: forecasts per season with and without wind data, by Study B split. |
| `reports/figures/fig23c_wind_by_split.png` | Figure: distributions of deep-layer wind and shear in the Study B training, validation and test seasons. |
| `reports/figures/fig23d_wind_correlations.png` | Figure: correlations among steering-wind features and persistence error. |
| `reports/figures/fig23e_wind_by_latitude.png` | Figure: deep-layer steering wind vs actual storm motion by latitude (recurvature and beta drift). |
| `reports/figures/fig23_steering_ring_katrina.png` | Figure: deep-layer steering flow and the 200–800 km ring around Hurricane Katrina, with persistence, steering and actual 24-h motion. |
| `reports/figures/fig24_steering_vs_motion.png` | Figure: steering displacement vs actual motion, and steering-minus-persistence vs persistence error. |
| `reports/figures/fig25_studyb_xgb_results.png` | Figure: Study B validation and test errors for persistence, steering only, ridge and XGBoost with/without winds. |
| `reports/figures/fig26_xgb_importance_with_winds.png` | Figure: XGBoost feature importance with steering features. |
| `reports/figures/fig27_studyb_test_results.png` | Figure: Study B test errors for every model with and without winds, all basins and North Atlantic. |
| `reports/figures/fig28_studyb_by_latitude.png` | Figure: Study B test error by latitude band, with and without winds. |
| `reports/figures/fig29_studyb_example_storm.png` | Figure: GRU 24-h forecasts with and without winds along Nadine (2012). |
| `reports/figures/fig30_final_test_results.png` | Figure: final test errors of every model, all basins and North Atlantic, vs NHC. |
| `reports/figures/fig31_validation_vs_test.png` | Figure: each model's validation error next to its test error. |
| `reports/figures/fig32_test_error_by_basin_latitude.png` | Figure: test error by basin and by latitude band (persistence, XGBoost, GRU round 2). |
| `reports/figures/fig33_test_error_by_intensity_season.png` | Figure: test error by intensity at forecast time and by season. |
| `reports/figures/fig34_test_error_cdf.png` | Figure: cumulative distribution of test errors. |
| `reports/figures/fig35_test_example_storm.png` | Figure: persistence, XGBoost and GRU 24-h forecasts along Franklin (2023). |
| `models/` | Saved models, not tracked: `xgb_dx.json`, `xgb_dy.json`, `xgb_config.json` (notebook 03); `lstm_seed*.pt`, `gru_seed*.pt`, `rnn_config.json` (notebook 04); `tuned_gru_seed*.pt`, `tuned_lstm_seed*.pt`, `tuned_config.json` (notebook 05); `round2_gru_seed*.pt`, `round2_config.json` (notebook 06); `studyb_gru_tracks_only_seed*.pt`, `studyb_gru_tracks_winds_seed*.pt`, `studyb_config.json` (notebook 07b). |
| `data/raw/` | Raw downloads (not tracked). |
| `data/processed/` | Notebook outputs, not tracked: `tracks_clean.csv.gz` (105,747 clean fixes), `cleaning_log.csv` (rows kept at each step), `eda_baselines.csv` (naive-forecast errors by split, from notebook 02), `preds_val_baselines.csv` (per-forecast validation errors and positions of each baseline, from notebook 03), `preds_val_rnn.csv` (per-forecast validation errors of the LSTM and GRU ensembles, from notebook 04), `preds_val_tuned.csv` and `tuning_log.csv` (tuned ensembles' errors and every tuning run, from notebook 05), `optuna_trials.csv` and `preds_val_round2.csv` (every Optuna trial and the round 2 ensemble's errors, from notebook 06), `steering_features.csv.gz` and `preds_studyb_xgb.csv` (Study B features, split and errors, from notebook 07a), `preds_studyb_gru.csv` (Study B GRU errors, from notebook 07b), `preds_test_baselines.csv` (baselines' test errors and positions, from 08a), `preds_test_all.csv` and `final_results_test.csv` (every model's test errors and the final table, from 08b). |
| `reports/notebook_pdfs/` | PDF export of every notebook for the course submission, one per notebook with the same name (`01_data_cleaning.pdf` … `08b_final_test_analysis.pdf`), made from the saved outputs by `scripts/export_notebook_pdfs.py`. |

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

## Study A: validation results (seasons 2017–2019)

| Model | Mean 24-h error | North Atlantic | Skill vs persistence |
|---|---|---|---|
| No motion | 391 km | 462 km | −133% |
| Persistence | 168 km | 183 km | 0% |
| Linear regression (CLIPER-style) | 158 km | 166 km | 5.8% |
| XGBoost (residual target) | 148.5 km | 164 km | 11.6% |
| LSTM (3-seed ensemble) | 146.0 km | 157 km | 13.0% |
| GRU (3-seed ensemble) | 145.2 km | 156 km | 13.5% |
| LSTM tuned (48-h history, 128 units) | 144.3 km | 155 km | 14.1% |
| GRU tuned, round 1 (48-h history, 128 units) | 144.0 km | 156 km | 14.3% |
| **GRU tuned, round 2 (Optuna: 48-h history, 256×2, dropout 0.3)** | **143.8 km** | **154 km** | **14.4%** |

## Study A: final results on the test seasons (2020–2024)

6,745 forecasts from 444 storms, never used for training or model choice. Every model was reloaded unchanged
and first shown to reproduce its validation score exactly.

| Model | Validation | **Test** | Test, North Atlantic | Test skill vs persistence |
|---|---|---|---|---|
| No motion | 391.1 km | 398.6 km | 461.7 km | −137.5% |
| Persistence | 168.0 km | 167.8 km | 179.1 km | 0% |
| Linear regression | 158.3 km | 159.8 km | 159.7 km | 4.7% |
| XGBoost | 148.5 km | 150.3 km | 155.1 km | 10.4% |
| LSTM | 146.0 km | 148.5 km | 151.5 km | 11.5% |
| GRU | 145.2 km | 147.8 km | 150.6 km | 11.9% |
| LSTM tuned | 144.3 km | 147.5 km | 150.5 km | 12.1% |
| GRU tuned, round 1 | 144.0 km | 146.9 km | 149.2 km | 12.5% |
| **GRU round 2 (chosen before testing)** | 143.8 km | **147.7 km** | **149.5 km** | **12.0%** |

- The chosen GRU beats XGBoost by 2.6 km (95% CI 1.3–3.8) and persistence by 20.1 km (17.4–22.8).
- The round 1 GRU beat it by 0.8 km (0.3–1.4): the small tuning gains on validation did not carry over. The chosen model is kept, since switching after seeing test results would be selecting on the test set.
- NHC's official Atlantic 24-hour error for 2020–2024 is about 63 km; the chosen GRU's North Atlantic error is about 2.4 times that. The worst forecasts are sudden stalls, loops and accelerations driven by the surrounding weather, which Study B shows is the information a track-only model lacks.

## Study B results: adding steering winds (test seasons 2009–2013)

Same 56,434 forecasts and split for every model; settings carried over unchanged from Study A. Errors are
not comparable with Study A's, since the seasons differ.

| Model | Mean 24-h error | North Atlantic | Skill vs persistence | Gain from winds (95% CI) |
|---|---|---|---|---|
| Persistence | 171.6 km | 183 km | 0% | |
| Steering wind only | 225.2 km | 247 km | −31.2% | |
| Linear regression, tracks only → + winds | 161.8 → 147.1 km | 163 → 145 km | 5.7% → 14.3% | 14.8 km (12.2–17.6) |
| XGBoost, tracks only → + winds | 152.1 → 138.1 km | 153 → 140 km | 11.4% → 19.5% | 14.0 km (11.2–17.0) |
| **GRU round 2, tracks only → + winds** | 149.0 → **136.2 km** | 151 → **139 km** | 13.2% → **20.7%** | 12.8 km (10.3–15.4) |

Adding steering winds improved every model by 13–15 km, several times more than tuning or the choice of model.
The gains are largest at 20–30° latitude, where storms recurve.

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In *Advances in Neural Information Processing Systems 24* (pp. 2546–2554).

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

Chan, J. C. L. (2005). The physics of tropical cyclone motion. *Annual Review of Fluid Mechanics, 37*, 99–128. https://doi.org/10.1146/annurev.fluid.37.061903.175702

Chan, J. C. L., & Gray, W. M. (1982). Tropical cyclone movement and surrounding flow relationships. *Monthly Weather Review, 110*(10), 1354–1374. https://doi.org/10.1175/1520-0493(1982)110<1354:TCMASF>2.0.CO;2

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 1724–1734). https://doi.org/10.3115/v1/D14-1179

Gahtan, J., Knapp, K. R., Schreck, C. J., Diamond, H. J., Kossin, J. P., & Kruk, M. C. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01*. NOAA National Centers for Environmental Information. https://doi.org/10.25921/82ty-9e16

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for assessing hyperparameter importance. In *Proceedings of the 31st International Conference on Machine Learning* (pp. 754–762).

Kalnay, E., Kanamitsu, M., Kistler, R., Collins, W., Deaven, D., Gandin, L., ... Joseph, D. (1996). The NCEP/NCAR 40-year reanalysis project. *Bulletin of the American Meteorological Society, 77*(3), 437–471. https://doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2

Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., & Neumann, C. J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society, 91*(3), 363–376.

Landsea, C. W., & Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new database format. *Monthly Weather Review, 141*(10), 3576–3592.

National Hurricane Center. (2025). *Official Atlantic basin annual average track errors, 1989–present* [Data set]. National Oceanic and Atmospheric Administration. https://www.nhc.noaa.gov/verification/pdfs/1989-present_OFCL_ATL_annual_trk_errors.pdf

Neumann, C. J. (1972). *An alternate to the HURRAN tropical cyclone forecast system* (NOAA Technical Memorandum NWS SR-62). National Weather Service.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

## AI-tool disclosure

Claude (Anthropic) was used to help with documentation in this repository. All code was supervised , reviewed, run, and checked by the author.
