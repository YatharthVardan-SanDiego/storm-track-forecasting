"""Record and draw the learning curve of the final (round 2) GRU, for the report's Results section.

Notebook 06 trained the final configuration but kept only the trained weights, not the per-epoch
history. So, here we train that same configuration once more, with seed 42 and exactly the same data
preparation (training seasons 1980-2016, early stopping on 2017-2019), record training and validation
error after every epoch, and draw them at the report's one-column size.

Outputs: data/processed/learning_curve_final.csv and reports/figures/report/r10_learning_curve_final.png
Usage:   conda activate stormtrack && python scripts/learning_curve_final.py   (about half a minute on MPS)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormtrack.data_io import load_clean, PROCESSED                                    # noqa: E402
from stormtrack.windows import build_samples, build_sequences, persistence_displacement, TARGET_COLS, STEP_FEATURES  # noqa: E402
from stormtrack.rnn import RNNForecaster, train_model                                    # noqa: E402

CFG = json.loads((ROOT / "models" / "round2_config.json").read_text())
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(4)

# So, first the same samples, static features and history length as the final model.
tracks = load_clean()
S = build_samples(tracks)
for b in sorted(S["BASIN"].unique()):
    S[f"basin_{b}"] = (S["BASIN"] == b).astype(float)
STATIC = CFG["static_features"]
is_tr, is_va = S["split"].eq("train").to_numpy(), S["split"].eq("val").to_numpy()
seq = build_sequences(tracks, S, n_steps=CFG["n_steps"])
Y = S[TARGET_COLS].to_numpy() - persistence_displacement(S)          # correction to persistence, km

# Then the scaling saved with the final model (computed on the training seasons only), so the
# inputs match notebook 06 exactly; padded history steps stay at zero.
mu, sd = np.array(CFG["seq_mu"]), np.array(CFG["seq_sd"])
seq_sc = np.where(seq[..., -1:] == 1, (seq - mu) / sd, 0.0).astype(np.float32)
st = ((S[STATIC].to_numpy() - np.array(CFG["st_mu"])) / np.array(CFG["st_sd"])).astype(np.float32)
y_sc = np.array(CFG["y_scale"])
tensors = lambda m: (torch.tensor(seq_sc[m]), torch.tensor(st[m]), torch.tensor(Y[m] / y_sc, dtype=torch.float32))

# Here, we train once with seed 42 and the final settings, keeping the per-epoch history.
torch.manual_seed(42)
model = RNNForecaster(len(STEP_FEATURES), len(STATIC), cell=CFG["cell"], hidden=CFG["hidden"],
                      layers=CFG["layers"], dropout=CFG["dropout"])
_, hist = train_model(model, tensors(is_tr), tensors(is_va), y_sc, DEVICE, seed=42, lr=CFG["lr"],
                      weight_decay=CFG["weight_decay"], batch_size=CFG["batch_size"], verbose=False)
h = pd.DataFrame(hist)
h.to_csv(PROCESSED / "learning_curve_final.csv", index=False)
best = h["val_km"].idxmin()
print(f"epochs {len(h)}, best epoch {h.loc[best, 'epoch']} at {h.loc[best, 'val_km']:.1f} km validation, "
      f"training {h.loc[best, 'train_km']:.1f} km; last epoch train {h['train_km'].iloc[-1]:.1f} / val {h['val_km'].iloc[-1]:.1f}")

# Finally, the figure: training and validation error per epoch, with the restored best epoch marked.
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.3, 1.95))
ax.plot(h["epoch"], h["train_km"], color="#2F6DB3", lw=1.2, label="Training (1980–2016)")
ax.plot(h["epoch"], h["val_km"], color="#C9453B", lw=1.2, label="Validation (2017–2019)")
ax.axvline(h.loc[best, "epoch"], color="k", ls=":", lw=0.8)
ax.scatter(h.loc[best, "epoch"], h.loc[best, "val_km"], color="k", s=10, zorder=3,
           label=f"Best epoch ({h.loc[best, 'val_km']:.1f} km)")
ax.set(xlabel="Epoch", ylabel="Mean distance error (km)")
ax.legend(frameon=False, loc="upper right")
out = ROOT / "reports" / "figures" / "report" / "r10_learning_curve_final.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out.relative_to(ROOT))
