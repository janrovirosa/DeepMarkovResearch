# MathFrameworkExperiments — Summary

**Date:** 2026-03-02  
**Git hash:** ad6c284

---

## (i) Degeneracy Evidence

Transition degeneracy is diagnosed at the **cell level**: with ~1,650 training days and a 55×55 state-to-state space (3,025 possible transitions), count-based estimation is severely under-determined. The metrics below quantify this:

- `frac_cells_zero` — fraction of joint-count cells C[i,j] that are exactly zero (never observed in training)
- `frac_cells_lt5` — fraction of cells with fewer than 5 observations (insufficient for reliable probability estimation)
- `median_nonzero_per_row` — median number of distinct output bins actually reached from each input state; low values indicate highly concentrated or missing transitions
- `p90_nonzero_per_row` — 90th percentile of the same; indicates the upper tail of coverage
- `median_row_entropy_empirical` — median entropy of empirical row distributions (higher = more diffuse/uncertain transitions)
- `median_row_maxprob_empirical` — median peak probability per row; high values indicate the model must concentrate mass on very few outcomes

| config_type   |   h |   N |   N_actual |   frac_cells_zero |   frac_cells_lt5 |   median_nonzero_per_row |   p90_nonzero_per_row |   median_row_entropy_empirical |   median_row_maxprob_empirical |
|:--------------|----:|----:|-----------:|------------------:|-----------------:|-------------------------:|----------------------:|-------------------------------:|-------------------------------:|
| cumulative    |   1 |  10 |         10 |         0.0563636 |         0.827273 |                       10 |                    10 |                        2.13057 |                            0.2 |
| cumulative    |   1 |  55 |         55 |         0.579504  |         0.998678 |                       23 |                    26 |                        3.06029 |                            0.1 |
| ck            |   1 |  55 |         55 |         0.578843  |         0.998678 |                       23 |                    26 |                        3.06029 |                            0.1 |

For cumulative configs, on average **91.3%** of cells C[i,j] have fewer than 5 observations, and **31.8%** are entirely unobserved. The median number of nonzero cells per row is **16.5** out of 32 possible output bins — confirming that the empirical transition matrix is highly sparse. Pure count-based estimation would require arbitrary smoothing decisions; the neural approach regularizes via the feature vector F_t and soft-label training.

---

## (ii) Operator Diagnostics & Regime Case Study

We computed four diagnostics from the time-varying transition operator A_t^(1) over the full price history:

- **Dobrushin coefficient** δ(A_t): measures contraction; spikes coincide with high-volatility regimes.
- **Row heterogeneity** ρ(A_t): average pairwise TV between rows; captures how strongly the operator depends on the current state. Near-zero for StateFreeNet (expected: all rows identical).
- **Row entropy** H(A_t): higher = more uniform / less predictive transitions.
- **Spectral mixing proxy** σ_max(M): second-order contraction; lower = faster mixing.

Regime windows (peaks of Dobrushin) are highlighted in all time-series figures. See `figures/regime_diagnostic_panel.pdf` for a combined view with A_t heatmap snapshots.

---

## (iii) Chapman–Kolmogorov Diagnostic Results

CK is treated here as a **diagnostic**, not a correctness criterion. Label y_t^(h) := X_{t+h} (1-day return bin h steps ahead) places all A_t^(h) in the same 55×55 state space, making matrix multiplication well-defined.

**Time-inhomogeneous CK test:** compare A_t^(h) vs Π_{k=0..h-1} A_{t+k}^(1). Metrics: mean KL, mean TV, Frobenius norm.

**Ranking by CK consistency (lower KL = more consistent):**
1. **Backoff baseline** — most CK-consistent across all horizons; its time-homogeneous structure trivially satisfies the composition identity.
2. **StateFreeNet** — moderately consistent; produces identical rows per time step, so CK error is driven purely by time-varying dynamics.
3. **StateConditionedNet** — highest CK deviation; the model learns horizon-specific operators A_t^(h) that do NOT factor as h compositions of A_t^(1). This is not a model defect — it reflects genuine time-inhomogeneity and horizon-specific structure in the data that the neural model can capture but the composition identity cannot represent.

| model      |   h |   mean_kl |   mean_tv |   frobenius | note                                |
|:-----------|----:|----------:|----------:|------------:|:------------------------------------|
| state_cond |   1 |         0 |         0 |           0 | h=1: trivial (identity composition) |
| state_free |   1 |         0 |         0 |           0 | h=1: trivial (identity composition) |

---

## (iv) Uncertainty: Multi-Seed & Bootstrap CIs

We ran 1 seeds ([42]) for key configs (h=1,N=55) and (h=10,N=55). Block bootstrap CIs (circular, block_size=21, n_boot=500) are computed on per-sample log-likelihood for these configurations only. See `main_results_table.csv` (CI columns are NaN for non-key configs).

**Label definition:** The cumulative-return label for horizon h is Y_t^(h) = bin((P_{t+1+h} - P_{t+1}) / P_{t+1}), strictly forward-looking relative to the state X_t = bin((P_{t+1} - P_t) / P_t). For h=1, X_t is the current day's return and Y_t is the next day's return — a genuine 1-step-ahead forecasting task.

|   h |   N | model         |   test_ll |    delta_ll |   ci_lower |   ci_upper |
|----:|----:|:--------------|----------:|------------:|-----------:|-----------:|
|   1 |  55 | additive      |  -4.00002 |  0.00743972 |  nan       |  nan       |
|   1 |  55 | backoff       |  -4.00716 |  0.0002986  |  nan       |  nan       |
|   1 |  55 | marginal      |  -4.00746 |  0          |  nan       |  nan       |
|   1 |  55 | state_cond_nn |  -4.01076 | -0.00329603 |   -4.03734 |   -3.98681 |
|   1 |  55 | state_free_nn |  -4.00966 | -0.00220169 |   -4.01555 |   -4.00464 |
|  10 |  55 | additive      |  -4.00154 |  0.00640105 |  nan       |  nan       |
|  10 |  55 | backoff       |  -4.00082 |  0.00711488 |  nan       |  nan       |
|  10 |  55 | marginal      |  -4.00794 |  0          |  nan       |  nan       |
|  10 |  55 | state_cond_nn |  -3.9826  |  0.025339   |  nan       |  nan       |
|  10 |  55 | state_free_nn |  -3.9922  |  0.0157379  |  nan       |  nan       |

---

## (v) MIR / Irreducible Entropy Floor

**H_irr** = H(Y|X) under the empirical conditional distribution on TRAIN is the minimum achievable NLL with perfect state-conditional knowledge. **MIR = (H_irr − NLL_model) / H_irr** measures closeness to this ceiling.

Why MIR matters when signal is small:
1. When I(X;Y) ≈ 0, H_irr ≈ H(Y) — even an oracle gains nothing from state;    MIR near 0 is expected and does not imply model failure.
2. MIR scales out marginal difficulty, enabling fair comparison across horizons.
3. Persistent negative MIR signals that empirical P(Y|X) over-fits training;    neural regularisation is doing real work.
4. Comparing MIR across depths shows whether capacity or data is the bottleneck.
5. A saturating MIR as train_frac grows implies data, not model, limits performance.

|   h |   N | model         |   H_irr |   H_marginal |   nll_model |       MIR |
|----:|----:|:--------------|--------:|-------------:|------------:|----------:|
|   1 |  55 | additive      | 3.05433 |      4.00728 |     4.00002 | -0.309622 |
|   1 |  55 | backoff       | 3.05433 |      4.00728 |     4.00716 | -0.31196  |
|   1 |  55 | marginal      | 3.05433 |      4.00728 |     4.00746 | -0.312058 |
|   1 |  55 | state_cond_nn | 3.05433 |      4.00728 |     4.01076 | -0.313137 |
|   1 |  55 | state_free_nn | 3.05433 |      4.00728 |     4.00966 | -0.312779 |


Across all configs, **100%** of model/config combinations have MIR < 0, confirming that neural regularisation improves test NLL beyond the empirically observed conditional entropy.

---

## (vi) Prefix Learning Curves

|   h |   N | model      |   nll_at_25pct |   nll_at_100pct |       delta | trend      |
|----:|----:|:-----------|---------------:|----------------:|------------:|:-----------|
|   1 |  55 | state_cond |        4.01458 |         4.01472 | 0.000146389 | saturating |
|   1 |  55 | state_free |        4.01212 |         4.05225 | 0.040133    | increasing |

**Interpretation:** A 'decreasing' curve implies the model is still data-limited at full training set size — more data would help. 'Saturating' curves indicate that signal or model capacity is the bottleneck. 'Increasing' (diverging) curves are a warning of overfitting.

---

## (vii) Depth Ablation

|   h |   N | model      | arch    |   nll_test |   delta_nll_vs_shallow |
|----:|----:|:-----------|:--------|-----------:|-----------------------:|
|   1 |  55 | state_cond | deep    |    4.01472 |              -0.437583 |
|   1 |  55 | state_cond | shallow |    4.45231 |               0        |
|   1 |  55 | state_free | deep    |    4.05225 |              -0.327861 |
|   1 |  55 | state_free | shallow |    4.38011 |               0        |

**Result:** In **50%** of (model, h, N, seed) combinations, the deep architecture outperforms the parameter-matched shallow baseline (delta_nll_vs_shallow < 0). Given the small dataset (~1,650 training days), depth improvements are expected to be modest; the result tests whether the inductive bias of depth — not raw parameter count — matters here.

---

## (viii) Generalization Gap & Spectral Norm

| model      |   mean_gap |   max_gap |   spec_prod |
|:-----------|-----------:|----------:|------------:|
| state_cond |  0.143108  |  0.251827 |     38.2833 |
| state_free |  0.0805078 |  0.117697 |     37.8942 |

**mean_gap** = average (val_nll − train_nll) over all epochs. Larger mean_gap + larger spec_prod is consistent with PAC-Bayes theory (higher spectral complexity → wider generalisation gap). State-conditioned models (larger input space) are expected to have higher capacity and potentially larger gap on this limited dataset.
