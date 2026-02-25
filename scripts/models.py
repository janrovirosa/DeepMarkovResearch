"""Model definitions.

StateConditionedNet and StateFreeNet are lifted verbatim from MasterNotebook.ipynb
(cell 23) with one change: constructors now accept `n_feat` as an explicit argument
instead of relying on a global `n_features` variable.  This allows reuse with any
feature dimensionality (194 for the master dataset).

GaussHeteroNet and MDNNet are optional lightweight continuous baselines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_HIDDEN = (64, 128, 256, 128, 64)
_DEFAULT_DROPOUT = 0.2


# ---------------------------------------------------------------------------
# Core models (lifted from master notebook, variable n_feat)
# ---------------------------------------------------------------------------

class StateConditionedNet(nn.Module):
    """[one_hot(X_t, N_XT); F_t] -> MLP -> n_output logits."""

    def __init__(
        self,
        n_feat: int,
        n_xt_states: int,
        n_output: int,
        hidden_dims: tuple = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.n_xt_states = n_xt_states
        layers = []
        in_dim = n_feat + n_xt_states
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, n_output)

    def forward(self, features: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        s_onehot = F.one_hot(x_t, num_classes=self.n_xt_states).float()
        z = torch.cat([s_onehot, features], dim=-1)
        return self.out(self.mlp(z))


class StateFreeNet(nn.Module):
    """F_t -> MLP -> n_output logits (no state conditioning)."""

    def __init__(
        self,
        n_feat: int,
        n_output: int,
        hidden_dims: tuple = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        layers = []
        in_dim = n_feat
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, n_output)

    def forward(self, features: torch.Tensor, x_t=None) -> torch.Tensor:
        return self.out(self.mlp(features))


# ---------------------------------------------------------------------------
# Optional continuous baselines
# ---------------------------------------------------------------------------

class GaussHeteroNet(nn.Module):
    """F_t -> (mu, log_sigma) for heteroskedastic Gaussian over continuous returns.

    For fair per-bin NLL comparison, discretize via CDF integration:
        P(bin_j) = Phi((e_{j+1} - mu) / sigma) - Phi((e_j - mu) / sigma)
    where Phi = scipy.stats.norm.cdf and e_0=-inf, e_N=+inf.
    """

    def __init__(
        self,
        n_feat: int,
        hidden_dims: tuple = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        layers = []
        in_dim = n_feat
        for h in hidden_dims[:3]:          # use first 3 layers only (lightweight)
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head_mu = nn.Linear(in_dim, 1)
        self.head_logsig = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor, x_t=None):
        h = self.mlp(features)
        mu = self.head_mu(h).squeeze(-1)
        log_sigma = self.head_logsig(h).squeeze(-1)
        return mu, log_sigma

    @staticmethod
    def gaussian_nll(mu, log_sigma, targets):
        """Gaussian NLL loss: 0.5 * [((y-mu)/sigma)^2 + log(sigma^2)]."""
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        return 0.5 * (((targets - mu) / sigma) ** 2 + 2 * log_sigma).mean()

    @staticmethod
    def discretize(mu_np, sigma_np, edges_np):
        """Integrate predicted Gaussian CDF over bin edges.

        Parameters
        ----------
        mu_np, sigma_np : np.ndarray, shape (T,)
        edges_np : np.ndarray, shape (N+1,)  with -inf/+inf boundaries

        Returns
        -------
        probs : np.ndarray, shape (T, N)  — rows sum to 1
        """
        from scipy.stats import norm
        import numpy as np
        N = len(edges_np) - 1
        T = len(mu_np)
        # CDF at each interior edge: shape (T, N+1)
        cdf = norm.cdf(
            edges_np[None, :],  # (1, N+1)
            loc=mu_np[:, None],
            scale=sigma_np[:, None],
        )
        probs = np.diff(cdf, axis=1)          # (T, N)
        probs = np.clip(probs, 1e-10, None)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


class MDNNet(nn.Module):
    """Mixture Density Network: F_t -> K Gaussian components.

    For fair per-bin NLL comparison, discretize via CDF integration.
    """

    def __init__(
        self,
        n_feat: int,
        K: int = 3,
        hidden_dims: tuple = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.K = K
        layers = []
        in_dim = n_feat
        for h in hidden_dims[:3]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head_pi = nn.Linear(in_dim, K)
        self.head_mu = nn.Linear(in_dim, K)
        self.head_logsig = nn.Linear(in_dim, K)

    def forward(self, features: torch.Tensor, x_t=None):
        h = self.mlp(features)
        pi = torch.softmax(self.head_pi(h), dim=-1)
        mu = self.head_mu(h)
        log_sigma = self.head_logsig(h)
        return pi, mu, log_sigma

    @staticmethod
    def mdn_nll(pi, mu, log_sigma, targets):
        """Negative log mixture likelihood."""
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        y = targets.unsqueeze(-1)              # (B, 1)
        log_prob = -0.5 * (((y - mu) / sigma) ** 2 + 2 * log_sigma)  # (B, K)
        log_prob -= 0.5 * torch.log(torch.tensor(2 * 3.14159265358979))
        log_mix = torch.log(pi + 1e-10) + log_prob                   # (B, K)
        return -torch.logsumexp(log_mix, dim=-1).mean()

    @staticmethod
    def discretize(pi_np, mu_np, sigma_np, edges_np):
        """Integrate predicted mixture CDF over bin edges.

        Parameters
        ----------
        pi_np    : np.ndarray, shape (T, K)
        mu_np    : np.ndarray, shape (T, K)
        sigma_np : np.ndarray, shape (T, K)
        edges_np : np.ndarray, shape (N+1,)

        Returns
        -------
        probs : np.ndarray, shape (T, N)
        """
        from scipy.stats import norm
        import numpy as np
        K = pi_np.shape[1]
        N = len(edges_np) - 1
        # CDF of each component at each edge: (T, K, N+1)
        cdf_k = norm.cdf(
            edges_np[None, None, :],
            loc=mu_np[:, :, None],
            scale=sigma_np[:, :, None],
        )
        prob_k = np.diff(cdf_k, axis=2)       # (T, K, N)
        probs = (pi_np[:, :, None] * prob_k).sum(axis=1)  # (T, N)
        probs = np.clip(probs, 1e-10, None)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
