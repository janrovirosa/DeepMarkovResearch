import copy
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


def dataset(F_, X, y, idx):
    return TensorDataset(torch.tensor(F_[idx], dtype=torch.float32),
                         torch.tensor(X[idx], dtype=torch.long),
                         torch.tensor(y[idx], dtype=torch.long))


def loaders(F_, X, y, sp, batch_train=256, batch_eval=512):
    # sp holds idx_train / idx_val / idx_test
    return (DataLoader(dataset(F_, X, y, sp["idx_train"]), batch_size=batch_train, shuffle=True),
            DataLoader(dataset(F_, X, y, sp["idx_val"]), batch_size=batch_eval),
            DataLoader(dataset(F_, X, y, sp["idx_test"]), batch_size=batch_eval))


def history_loaders(F_, X, y, sp, k, batch_train=256, batch_eval=512):
    # each sample carries [X_t, X_{t-1}, ..., X_{t-k+1}], dropping indices without full history
    def ds(idx):
        idx = idx[idx >= k - 1]
        hist = np.stack([X[idx - lag] for lag in range(k)], axis=1)
        return TensorDataset(torch.tensor(F_[idx], dtype=torch.float32),
                             torch.tensor(hist, dtype=torch.long),
                             torch.tensor(y[idx], dtype=torch.long))
    return (DataLoader(ds(sp["idx_train"]), batch_size=batch_train, shuffle=True),
            DataLoader(ds(sp["idx_val"]), batch_size=batch_eval),
            DataLoader(ds(sp["idx_test"]), batch_size=batch_eval))


def soft_labels(y, n, sigma):
    # gaussian kernel over neighbouring bins instead of a one-hot target
    j = torch.arange(n, device=y.device, dtype=torch.float32).unsqueeze(0).expand(len(y), -1)
    w = torch.exp(-((j - y.unsqueeze(1).float()) ** 2) / (2 * sigma ** 2))
    return w / (w.sum(dim=1, keepdim=True) + 1e-8)


def kl_loss(logits, y, n, sigma):
    return F.kl_div(F.log_softmax(logits, dim=1), soft_labels(y, n, sigma), reduction="batchmean")


def nll(model, loader, device):
    total, count = 0.0, 0
    with torch.no_grad():
        for f, x, y in loader:
            f, x, y = f.to(device), x.to(device), y.to(device)
            total += F.cross_entropy(model(f, x), y, reduction="sum").item()
            count += len(y)
    return total / count


def train(model, train_loader, val_loader, n, sigma, device, lr=1e-3, weight_decay=1e-4,
          max_epochs=100, patience=10, grad_clip=1.0, verbose=False, log_nll=False,
          swa=False, swa_epochs=10):
    # AdamW + plateau scheduler, early stopping on val loss, optional average of the last swa_epochs
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if log_nll:
        hist["train_nll"], hist["val_nll"] = [], []
    best, best_state, wait = float("inf"), None, 0
    last = deque(maxlen=swa_epochs)

    for epoch in range(1, max_epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for f, x, y in train_loader:
            f, x, y = f.to(device), x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(f, x)
            loss = kl_loss(logits, y, n, sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            tl += loss.item() * len(y)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(y)

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for f, x, y in val_loader:
                f, x, y = f.to(device), x.to(device), y.to(device)
                logits = model(f, x)
                vl += kl_loss(logits, y, n, sigma).item() * len(y)
                vc += (logits.argmax(1) == y).sum().item()
                vn += len(y)

        tl, vl = tl / tn, vl / vn
        sched.step(vl)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(tc / tn)
        hist["val_acc"].append(vc / vn)
        if log_nll:
            hist["train_nll"].append(nll(model, train_loader, device))
            hist["val_nll"].append(nll(model, val_loader, device))

        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if swa:
            last.append(copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"  epoch {epoch:>3}: train={tl:.4f} val={vl:.4f} t_acc={tc/tn:.3f} v_acc={vc/vn:.3f}")
        if wait >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if swa:
        best_state = {k: torch.stack([s[k].float() for s in last]).mean(0).to(last[0][k].dtype)
                      for k in last[0]}
        print(f"[SWA] Averaged {len(last)} checkpoints")
    model.load_state_dict(best_state)
    model.to(device)
    return best_state, hist


def fit(model, path, train_loader, val_loader, n, sigma, device, **kw):
    # load cached weights if they exist, otherwise train and cache them
    if path is not None and Path(path).exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.to(device), None
    state, hist = train(model, train_loader, val_loader, n, sigma, device, **kw)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
    return model, hist


def operators(model, F_, t_idx, n_states, n_out, device, state_free=False):
    # A[t, i, :] = P(next | X_t = i, F_t), one forward pass per t over all states
    model.eval()
    A = np.zeros((len(t_idx), n_states, n_out), dtype=np.float32)
    states = torch.arange(n_states, dtype=torch.long, device=device)
    with torch.no_grad():
        for pos, t in enumerate(t_idx):
            f = torch.tensor(F_[t], dtype=torch.float32).unsqueeze(0)
            if state_free:
                A[pos] = F.softmax(model(f.to(device), states[:1]), dim=1).cpu().numpy()[0][None, :]
            else:
                A[pos] = F.softmax(model(f.expand(n_states, -1).to(device), states), dim=1).cpu().numpy()
    return A
