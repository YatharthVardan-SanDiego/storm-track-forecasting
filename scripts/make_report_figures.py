"""Draw the figures used in the final report, sized for one column of the report template.

The notebooks' figures are drawn for a full-width screen; the report's two-column layout
leaves about 3.3 inches per figure, where those would be unreadable. This script redraws
the main results at that size from the files the notebooks saved, so every number in a
report figure comes from the same outputs as the notebooks.

Needs: notebooks 01-08b already run (data/processed/*), and NCEP winds for 2005 (Figure R6).
Writes PNGs (300 dpi) to reports/figures/report/.

Usage:
    conda activate stormtrack
    python scripts/make_report_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormtrack.data_io import load_clean, PROCESSED          # noqa: E402
from stormtrack.windows import build_samples, persistence_displacement, track_error_km   # noqa: E402
from stormtrack.steering import NCEP_DIR, RING_KM               # noqa: E402

OUT = ROOT / "reports" / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)
W = 3.3                                                          # inches: one column of the report
KW = dict(keep_default_na=False, na_values=[""], parse_dates=["ISO_TIME"])   # keep basin code "NA"
COL = {"Persistence": "#E07B39", "XGBoost": "#2F6DB3", "GRU": "#C9453B", "Ridge": "#8064A2", "steer": "#2F9E44"}
BASIN_COL = {"WP": "#1b9e77", "SI": "#d95f02", "EP": "#7570b3", "NA": "#e7298a", "SP": "#66a61e", "NI": "#e6ab02", "SA": "#666666"}
NHC_KM = 63

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.2, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", (OUT / name).relative_to(ROOT))


tracks = load_clean()
S = build_samples(tracks)

# --- R1. Every cleaned track, coloured by basin ---------------------------------------------
fig, ax = plt.subplots(figsize=(W, 1.75))
for basin, g in tracks.groupby("BASIN"):
    for _, seg in g.groupby("SEG_ID"):
        lon, lat = seg["LON"].to_numpy() % 360, seg["LAT"].to_numpy().astype(float)   # centre on the Pacific
        jump = np.r_[False, np.abs(np.diff(lon)) > 180]       # break the line where a track wraps past 0°
        lon, lat = np.insert(lon, np.where(jump)[0], np.nan), np.insert(lat, np.where(jump)[0], np.nan)
        ax.plot(lon, lat, color=BASIN_COL[basin], lw=0.15, alpha=0.35)
ax.set(xlim=(25, 360), ylim=(-50, 55), xlabel="Longitude (°E)", ylabel="Latitude")
ax.set_xticks([60, 120, 180, 240, 300, 360], ["60", "120", "180", "240", "300", "360"])
ax.legend(handles=[Line2D([], [], color=c, lw=1.5, label=b) for b, c in BASIN_COL.items() if b != "SA"],
          ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, handlelength=1.2, columnspacing=0.8)
save(fig, "r1_track_map.png")

# --- R2. Persistence breaks down where storms recurve ----------------------------------------
P = persistence_displacement(S)
S["err_persist"] = track_error_km(S, P)
S["band"] = (S["LAT"].abs() // 5 * 5).clip(upper=40).astype(int)
S["east"] = S["dx_target"] > 0
g = S.groupby("band").agg(err=("err_persist", "mean"), east=("east", "mean"), n=("err_persist", "size"))
g = g[g["n"] >= 200]
labels = [f"{b}–{b + 5}" if b < 40 else "40+" for b in g.index]
fig, ax = plt.subplots(figsize=(W, 1.9))
ax.bar(labels, g["err"], color=COL["Persistence"], alpha=0.85, label="Persistence 24-h error (km, left)")
ax.set(xlabel="Distance from the equator at forecast time (° latitude)", ylabel="Mean 24-h error (km)")
ax2 = ax.twinx()
ax2.plot(labels, 100 * g["east"], "o-", color="k", ms=2.5, label="Moving east over next 24 h (%, right)")
ax2.set(ylim=(0, 100), ylabel="Share moving east (%)")
ax2.spines["right"].set_visible(True)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
save(fig, "r2_persistence_by_latitude.png")

# --- R3. Architecture of the forecasting network ---------------------------------------------
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.set(xlim=(0, 10), ylim=(0, 11))
ax.axis("off")

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15", fc=fc, ec="#444", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.8, linespacing=1.25)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#444"))

box(0.2, 9.0, 4.6, 1.7, "Track history\n7 six-hour steps × 5 features\n(motion, latitude, wind, flag)", "#FCE9DD")
box(0.2, 6.9, 4.6, 1.2, "GRU layer 1: 256 units", "#F6D2CE")
box(0.2, 4.8, 4.6, 1.2, "GRU layer 2: 256 units", "#F6D2CE")
box(5.4, 9.0, 4.4, 1.7, "Static features (12)\nlongitude, day of year,\nland distance, basin", "#DCE6F4")
box(5.4, 4.8, 4.4, 1.2, "Concatenate: 268", "#EEEEEE")
box(2.6, 2.7, 4.8, 1.2, "Dense 256, ReLU, dropout 0.3", "#E4E0F0")
box(2.6, 0.3, 4.8, 1.5, "Dense 2: correction (km east,\nkm north) + persistence", "#E4E0F0")
arrow(2.5, 9.0, 2.5, 8.1)
arrow(2.5, 6.9, 2.5, 6.0)
arrow(4.8, 5.4, 5.4, 5.4)
arrow(7.6, 9.0, 7.6, 6.0)
arrow(7.6, 4.8, 6.2, 3.9)
arrow(5.0, 2.7, 5.0, 1.8)
ax.text(4.9, 6.1, "last hidden\nstate", fontsize=5, color="#555", va="bottom")
ax.text(2.7, 6.45, "dropout 0.3", fontsize=5, color="#555", va="center")
save(fig, "r3_architecture.png")

# --- R4. Final test results ------------------------------------------------------------------
final = pd.read_csv(PROCESSED / "final_results_test.csv", index_col=0)
order = ["Persistence", "Linear regression", "XGBoost", "LSTM", "GRU", "LSTM tuned", "GRU tuned", "GRU round 2"]
names = ["Persistence", "Linear regression", "XGBoost", "LSTM", "GRU", "LSTM, tuned (round 1)",
         "GRU, tuned (round 1)", "GRU, round 2 (chosen)"]
cols = [COL["Persistence"], COL["Ridge"], COL["XGBoost"]] + ["#E9A19B"] * 4 + [COL["GRU"]]
fig, ax = plt.subplots(figsize=(W, 2.6))
y = np.arange(len(order))[::-1]
ax.barh(y + 0.19, final.loc[order, "test_mean_km"], 0.38, color=cols, edgecolor="k", lw=0.3)
ax.barh(y - 0.19, final.loc[order, "test_NA_km"], 0.38, color=cols, alpha=0.45, edgecolor="k", lw=0.3)
for yi, a, b in zip(y, final.loc[order, "test_mean_km"], final.loc[order, "test_NA_km"]):
    ax.text(a + 2, yi + 0.19, f"{a:.1f}", va="center", fontsize=5.6)
    ax.text(b + 2, yi - 0.19, f"{b:.1f}", va="center", fontsize=5.6, color="#555")
ax.axvline(NHC_KM, color="k", ls="--", lw=0.8)
ax.set_yticks(y, names)
ax.set(xlim=(0, 200), xlabel="Mean 24-h track error, test seasons 2020–2024 (km)")
ax.legend(handles=[Patch(facecolor="#999", label="All basins"), Patch(facecolor="#999", alpha=0.45, label="North Atlantic"),
                   Line2D([], [], color="k", ls="--", lw=0.8, label="NHC official, Atlantic (63 km)")],
          loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=3, frameon=False, handlelength=1.5, columnspacing=0.8)
save(fig, "r4_final_test_results.png")

# --- R5. Test error by latitude --------------------------------------------------------------
pt = pd.read_csv(PROCESSED / "preds_test_all.csv", **KW)
pt["band"] = (pt["LAT"].abs() // 5 * 5).astype(int)
show = {"Persistence": ("err_persistence_km", COL["Persistence"]), "XGBoost": ("err_xgboost_km", COL["XGBoost"]),
        "GRU, round 2": ("err_gru_round_2_km", COL["GRU"])}
gb = pt.groupby("band").agg(**{k: (c, "mean") for k, (c, _) in show.items()}, n=("LAT", "size"))
gb = gb[gb["n"] >= 50]
lab = [f"{b}–{b + 5}" for b in gb.index]
fig, ax = plt.subplots(figsize=(W, 1.9))
for k, (_, c) in show.items():
    ax.plot(lab, gb[k], "o-", color=c, ms=2.5, label=k)
for i, n in enumerate(gb["n"]):
    ax.text(i, 104, f"n={n}", ha="center", fontsize=5.2, color="#555")
ax.set(ylim=(98, 285), xlabel="Distance from the equator at forecast time (° latitude)", ylabel="Mean 24-h error (km)")
ax.legend(frameon=False, loc="upper left")
save(fig, "r5_test_error_by_latitude.png")

# --- R6. Steering flow around Katrina, 28 Aug 2005 12 UTC -----------------------------------
steer = pd.read_csv(PROCESSED / "steering_features.csv.gz", **KW)
kat_sid = tracks.loc[tracks["NAME"].eq("KATRINA") & tracks["SEASON"].eq(2005) & tracks["BASIN"].eq("NA"), "SID"].iloc[0]
t0 = pd.Timestamp("2005-08-28 12:00")
row = S[(S["SID"] == kat_sid) & (S["ISO_TIME"] == t0)].merge(steer[["SEG_ID", "ISO_TIME", "steer_dx", "steer_dy"]],
                                                              on=["SEG_ID", "ISO_TIME"]).iloc[0]
with xr.open_dataset(NCEP_DIR / "ncep_winds_2005.nc") as ds:
    bx = ds.sel(time=t0, lat=slice(row["LAT"] + 12.5, row["LAT"] - 12.5),
                lon=slice(row["LON"] % 360 - 15, row["LON"] % 360 + 15)).load()
u, v = bx["uwnd"].mean("level"), bx["vwnd"].mean("level")
glon = bx["lon"].values - 360
fig, ax = plt.subplots(figsize=(W, 2.75))
q = ax.quiver(glon, bx["lat"], u, v, np.hypot(u, v), cmap="Blues", scale=220, width=0.005, clim=(0, 16))
cb = fig.colorbar(q, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Deep-layer mean wind (m/s)", fontsize=6)
cb.ax.tick_params(labelsize=5.5)
trk = tracks[tracks["SID"] == kat_sid]
ax.plot(trk["LON"], trk["LAT"], "k.-", lw=0.7, ms=1.5, label="Katrina's track")
th = np.linspace(0, 2 * np.pi, 200)
kmlon = 111.195 * np.cos(np.radians(row["LAT"]))
for r in RING_KM:
    ax.plot(row["LON"] + r / kmlon * np.cos(th), row["LAT"] + r / 111.195 * np.sin(th), "k--", lw=0.6)
for (dx, dy), c, l in [((4 * row["dx_m0"], 4 * row["dy_m0"]), COL["Persistence"], "Persistence"),
                       ((row["steer_dx"], row["steer_dy"]), COL["steer"], "Steering wind alone"),
                       ((row["dx_target"], row["dy_target"]), "k", "Actual")]:
    ax.annotate("", xy=(row["LON"] + dx / kmlon, row["LAT"] + dy / 111.195), xytext=(row["LON"], row["LAT"]),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    ax.plot([], [], color=c, lw=1.4, label=f"{l}, 24 h")
ax.set(xlim=(glon.min(), glon.max()), ylim=(row["LAT"] - 12.5, row["LAT"] + 12.5), aspect="equal",
       xlabel="Longitude", ylabel="Latitude")
ax.legend(loc="lower left", framealpha=0.9)
save(fig, "r6_steering_katrina.png")

# --- R7. Study B: with and without winds on the Study B test seasons -------------------------
xb = pd.read_csv(PROCESSED / "preds_studyb_xgb.csv", **KW)
gb_ = pd.read_csv(PROCESSED / "preds_studyb_gru.csv", **KW)
xt, gt = xb[xb["split_b"] == "test"], gb_[gb_["split_b"] == "test"]
pairs = {"Linear\nregression": (xt["err_ridge_tracks_only_km"], xt["err_ridge_tracks_winds_km"], COL["Ridge"]),
         "XGBoost": (xt["err_xgboost_tracks_only_km"], xt["err_xgboost_tracks_winds_km"], COL["XGBoost"]),
         "GRU\n(round 2)": (gt["err_gru_tracks_only_km"], gt["err_gru_tracks_winds_km"], COL["GRU"])}
fig, ax = plt.subplots(figsize=(W, 1.95))
x = np.arange(len(pairs))
for i, (name, (a, b, c)) in enumerate(pairs.items()):
    ax.bar(i - 0.19, a.mean(), 0.36, color=c, alpha=0.4, edgecolor="k", lw=0.3)
    ax.bar(i + 0.19, b.mean(), 0.36, color=c, edgecolor="k", lw=0.3)
    ax.text(i - 0.19, a.mean() + 2, f"{a.mean():.1f}", ha="center", fontsize=5.6)
    ax.text(i + 0.19, b.mean() + 2, f"{b.mean():.1f}", ha="center", fontsize=5.6)
ax.axhline(xt["err_persistence_km"].mean(), color=COL["Persistence"], ls="--", lw=0.8)
ax.set_xticks(x, list(pairs))
ax.set(ylim=(0, 190), ylabel="Mean 24-h error (km)")
ax.legend(handles=[Patch(facecolor="#999", alpha=0.4, label="Track features only"), Patch(facecolor="#999", label="Track + steering winds"),
                   Line2D([], [], color=COL["Persistence"], ls="--", lw=0.8, label=f"Persistence ({xt['err_persistence_km'].mean():.1f} km)")],
          loc="lower right", framealpha=0.9)
save(fig, "r7_studyb_test_results.png")

# --- R8. How long storm motion stays persistent (for the Data Summary) ------------------------
tr_sorted = tracks.sort_values(["SEG_ID", "ISO_TIME"])
seg = tr_sorted.groupby("SEG_ID")
lags = range(1, 13)
ac = {"East–west motion": [], "North–south motion": []}
for k in lags:
    for name, col in (("East–west motion", "dx_km"), ("North–south motion", "dy_km")):
        a, b = tr_sorted[col], seg[col].shift(-k)
        ok = a.notna() & b.notna()
        ac[name].append(np.corrcoef(a[ok], b[ok])[0, 1])
fig, ax = plt.subplots(figsize=(W, 1.85))
hours = [6 * k for k in lags]
for (name, vals), c in zip(ac.items(), ["#2F6DB3", "#C9453B"]):
    ax.plot(hours, vals, "o-", color=c, ms=2.5, label=name)
ax.axvline(24, color="#999", ls=":", lw=0.8)
ax.set(xlabel="Time between the two 6-hour motions (hours)", ylabel="Correlation", ylim=(0, 1), xticks=hours[1::2])
ax.legend(frameon=False, loc="upper right")
save(fig, "r8_motion_autocorrelation.png")

# --- R9. Steering wind vs storm motion by latitude (second-dataset EDA) -----------------------
from stormtrack.steering import STEERING_COLS, KM_PER_24H                         # noqa: E402
sb = S.merge(steer[["SEG_ID", "ISO_TIME", "split_b"] + [c for c in STEERING_COLS if c not in ("steer_dx", "steer_dy")]],
             on=["SEG_ID", "ISO_TIME"])
sb = sb[sb["split_b"] == "train"].copy()
sb["band9"] = (sb["LAT"].abs() // 5 * 5).astype(int)
pole = np.sign(sb["LAT"])
sb["mot_e"], sb["mot_p"] = sb["dx_target"] / KM_PER_24H, pole * sb["dy_target"] / KM_PER_24H
sb["st_p"] = pole * sb["dlm_v"]
prof = sb.groupby("band9").agg(n=("LAT", "size"), st_e=("dlm_u", "mean"), mot_e=("mot_e", "mean"),
                               st_p=("st_p", "mean"), mot_p=("mot_p", "mean"))
prof = prof[prof["n"] >= 300]
lab9 = [f"{b}–{b + 5}" for b in prof.index]
fig, ax = plt.subplots(figsize=(W, 2.1))
ax.plot(lab9, prof["st_e"], "o-", color=COL["steer"], ms=2.5, label="Steering wind, eastward")
ax.plot(lab9, prof["mot_e"], "s--", color="k", ms=2.5, label="Storm motion, eastward")
ax.plot(lab9, prof["st_p"], "o-", color=COL["steer"], alpha=0.45, ms=2.5, label="Steering wind, poleward")
ax.plot(lab9, prof["mot_p"], "s--", color="#777", ms=2.5, label="Storm motion, poleward")
ax.axhline(0, color="#999", lw=0.6)
ax.set(xlabel="Distance from the equator (° latitude)", ylabel="Speed (m/s)")
ax.legend(frameon=False, loc="upper left", ncol=1)
save(fig, "r9_wind_vs_motion_by_latitude.png")

# --- R11. Round 2 (Optuna) search history -------------------------------------------------------
trials = pd.read_csv(PROCESSED / "optuna_trials.csv")
done = trials[trials["state"] == "COMPLETE"]
pruned = trials[trials["state"] == "PRUNED"]
best_so_far = trials["value"].where(trials["state"] == "COMPLETE").cummin().ffill()   # carry through pruned trials
fig, ax = plt.subplots(figsize=(W, 1.95))
ax.scatter(done["number"], done["value"], color=COL["XGBoost"], s=9, zorder=3, label=f"Completed trial ({len(done)})")
ymin = done["value"].min() - 0.6
ax.scatter(pruned["number"], np.full(len(pruned), ymin + 0.15), marker="|", color="#999", s=30,
           label=f"Pruned trial ({len(pruned)})")
ax.step(trials["number"], best_so_far, where="post", color=COL["GRU"], lw=1.1, label="Best so far")
ax.axhline(trials.loc[trials["number"] == 0, "value"].iloc[0], color="k", ls="--", lw=0.7,
           label="Round 1 configuration (trial 0)")
ax.set(xlabel="Trial", ylabel="Error, 2014–2016 (km)", ylim=(ymin, done["value"].quantile(0.95) + 0.4))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, columnspacing=0.8)
save(fig, "r11_optuna_history.png")
