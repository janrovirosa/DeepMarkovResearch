import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = (64, 128, 256, 128, 64)


def mlp(n_in, hidden, dropout):
    layers = []
    for h in hidden:
        layers += [nn.Linear(n_in, h), nn.GELU(), nn.Dropout(dropout)]
        n_in = h
    return nn.Sequential(*layers), n_in


def one_hot(x, n, device):
    # built on cpu first because DirectML can't scatter
    return F.one_hot(x.cpu(), num_classes=n).float().to(device)


class StateCondNet(nn.Module):
    # [one_hot(X_t), F_t] -> logits over the next state / return bin
    def __init__(self, n_feat, n_states, n_out, hidden=HIDDEN, dropout=0.2):
        super().__init__()
        self.n_states = n_states
        self.mlp, n = mlp(n_feat + n_states, hidden, dropout)
        self.out = nn.Linear(n, n_out)

    def forward(self, f, x):
        return self.out(self.mlp(torch.cat([one_hot(x, self.n_states, f.device), f], dim=-1)))


class StateFreeNet(nn.Module):
    # F_t -> logits, ignores the state, so every row of A_t is the same
    def __init__(self, n_feat, n_out, hidden=HIDDEN, dropout=0.2):
        super().__init__()
        self.mlp, n = mlp(n_feat, hidden, dropout)
        self.out = nn.Linear(n, n_out)

    def forward(self, f, x=None):
        return self.out(self.mlp(f))


class HigherOrderNet(nn.Module):
    # [one_hot(X_t), ..., one_hot(X_{t-k+1}), F_t] -> logits, k=1 is StateCondNet
    def __init__(self, n_feat, n_states, n_out, k=1, hidden=HIDDEN, dropout=0.2):
        super().__init__()
        self.n_states, self.k = n_states, k
        self.mlp, n = mlp(k * n_states + n_feat, hidden, dropout)
        self.out = nn.Linear(n, n_out)

    def forward(self, f, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        z = [one_hot(x[:, i], self.n_states, f.device) for i in range(self.k)]
        return self.out(self.mlp(torch.cat(z + [f], dim=-1)))


def make_model(kind, n_feat, n_states, n_out, hidden=HIDDEN, dropout=0.2):
    if kind == "state_free":
        return StateFreeNet(n_feat, n_out, hidden, dropout)
    return StateCondNet(n_feat, n_states, n_out, hidden, dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shallow_width(target, **kw):
    # binary search for the one-layer width with the closest parameter count
    lo, hi = 1, 4096
    best, best_diff = lo, float("inf")
    for _ in range(20):
        mid = (lo + hi) // 2
        p = count_params(make_model(hidden=(mid,), **kw))
        if abs(p - target) < best_diff:
            best, best_diff = mid, abs(p - target)
        if p < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best
