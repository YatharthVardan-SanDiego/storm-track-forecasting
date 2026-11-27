"""Recurrent track-forecast model (LSTM or GRU) and its training loop, in PyTorch."""
import copy
import time

import numpy as np
import torch
from torch import nn


class RNNForecaster(nn.Module):
    """Sequence encoder + static context + small MLP head.

    The recurrent layer (LSTM or GRU) reads the storm's recent history one 6-hour
    step at a time. Its final hidden state, a summary of how the storm has been
    moving, is joined with features that do not change along the window (location,
    season, land distance, basin), and a two-layer MLP turns that into a 24-hour
    correction to the persistence forecast (east and north, scaled units).
    """

    def __init__(self, n_step_features, n_static, cell="lstm", hidden=64, layers=1, dropout=0.1):
        super().__init__()
        # So, first the recurrent encoder: an LSTM or GRU that reads the history one 6-hour step at a time.
        # With more than one layer, dropout is applied between the layers.
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(n_step_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        # Then the head: the encoder's summary plus the static features go through one ReLU layer with
        # dropout, and a linear layer gives the 2 outputs (east and north correction to persistence).
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, seq, static):
        # Here, we run the encoder over the whole sequence and keep only its final hidden state, a
        # summary of how the storm has been moving, then join it with the static features.
        _, h = self.rnn(seq)
        h_last = (h[0] if isinstance(h, tuple) else h)[-1]      # LSTM returns (h, c); GRU returns h
        return self.head(torch.cat([h_last, static], dim=1))


def distance_loss(pred_scaled, target_scaled, y_scale):
    """Mean Euclidean distance in km between predicted and true displacement.

    Predictions and targets are in scaled units; multiplying back by y_scale gives km,
    so the loss is (very nearly) the great-circle track error the models are judged on.
    """
    # Back to km first, so the loss is a real distance, then the straight-line length of each error.
    diff = (pred_scaled - target_scaled) * y_scale
    return torch.sqrt((diff ** 2).sum(dim=1) + 1e-6).mean()


def mse_loss(pred_scaled, target_scaled, y_scale=None):
    """Plain mean squared error in scaled units (the usual regression loss), for comparison."""
    return ((pred_scaled - target_scaled) ** 2).mean()


LOSSES = {"distance": distance_loss, "mse": mse_loss}


def train_model(model, train_tensors, val_tensors, y_scale, device, epochs=80, batch_size=512,
                lr=1e-3, weight_decay=1e-4, patience=10, seed=42, verbose=True, loss="distance",
                on_epoch=None):
    """Train with AdamW and early stopping; return the best model and the per-epoch history.

    `loss` picks what the optimizer minimizes ("distance" or "mse"). Whatever the choice,
    early stopping and the reported numbers always use the mean distance error in km on
    the validation set, so runs with different losses are judged the same way.

    `on_epoch(epoch, val_km)`, if given, is called after every epoch. A hyperparameter
    search uses it to report progress and to stop unpromising runs early (pruning).
    """
    # First, we set everything up: the loss to minimize, fixed seeds, the data on the device, AdamW, and
    # a scheduler that halves the learning rate after 4 epochs without improvement.
    loss_fn = LOSSES[loss]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = model.to(device)
    y_scale = torch.as_tensor(y_scale, dtype=torch.float32, device=device)
    Xs_tr, Xc_tr, Y_tr = [t.to(device) for t in train_tensors]
    Xs_va, Xc_va, Y_va = [t.to(device) for t in val_tensors]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    # Then the training loop. Within each epoch, the training cases are reshuffled and fed in batches.
    best, best_state, wait, hist = np.inf, None, 0, []
    n = len(Y_tr)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            # For each batch: forward pass, loss, backpropagation, gradient clipping (norm 1.0, against
            # exploding gradients in recurrent networks), and an optimizer step.
            opt.zero_grad()
            out = model(Xs_tr[idx], Xc_tr[idx])
            loss = loss_fn(out, Y_tr[idx], y_scale)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():                      # always report training error in km
                total += distance_loss(out, Y_tr[idx], y_scale).item() * len(idx)
        # After every epoch, we measure the mean distance error on the validation set and let the
        # scheduler (and, during a search, the pruner) react to it.
        model.eval()
        with torch.no_grad():
            val = distance_loss(model(Xs_va, Xc_va), Y_va, y_scale).item()
        sched.step(val)
        if on_epoch is not None:
            on_epoch(epoch, val)
        hist.append({"epoch": epoch, "train_km": total / n, "val_km": val,
                     "lr": opt.param_groups[0]["lr"], "secs": time.time() - t0})
        # Early stopping: keep a copy of the best weights so far, and stop once the validation error has
        # not improved by at least 0.01 km for `patience` epochs.
        if val < best - 0.01:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}  train {total / n:6.1f} km  val {val:6.1f} km")
        if wait >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}; best val {best:.1f} km")
            break
    # Finally, we restore the weights from the best epoch before returning.
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, seq, static, device, batch_size=4096):
    model.eval()
    out = [model(seq[i:i + batch_size].to(device), static[i:i + batch_size].to(device)).cpu()
           for i in range(0, len(seq), batch_size)]
    return torch.cat(out).numpy()
