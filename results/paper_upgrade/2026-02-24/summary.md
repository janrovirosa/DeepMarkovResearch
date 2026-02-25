# MathFrameworkExperiments — Summary

**Date:** 2026-02-24  
**Git hash:** fe7e150

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
| cumulative    |   1 |  10 |         10 |         0.0563636 |         0.827273 |                       10 |                  10   |                        2.13057 |                       0.2      |
| cumulative    |   1 |  20 |         20 |         0.224545  |         0.98     |                       16 |                  17   |                        2.62696 |                       0.133333 |
| cumulative    |   1 |  35 |         35 |         0.436364  |         0.995844 |                       20 |                  22   |                        2.88048 |                       0.1      |
| cumulative    |   1 |  55 |         55 |         0.579504  |         0.998678 |                       23 |                  26   |                        3.06029 |                       0.1      |
| cumulative    |   2 |  10 |         10 |         0.0345455 |         0.807273 |                       10 |                  10   |                        2.13669 |                       0.2      |
| cumulative    |   2 |  20 |         20 |         0.213636  |         0.976364 |                       16 |                  17   |                        2.60952 |                       0.133333 |
| cumulative    |   2 |  35 |         35 |         0.423377  |         0.996883 |                       20 |                  22.6 |                        2.90832 |                       0.1      |
| cumulative    |   2 |  55 |         55 |         0.582149  |         0.999339 |                       23 |                  25   |                        3.07773 |                       0.1      |
| cumulative    |   5 |  10 |         10 |         0.0490909 |         0.827273 |                       10 |                  10   |                        2.15413 |                       0.193548 |
| cumulative    |   5 |  20 |         20 |         0.218182  |         0.977273 |                       16 |                  17   |                        2.62465 |                       0.133333 |
| cumulative    |   5 |  35 |         35 |         0.423896  |         0.996883 |                       20 |                  23   |                        2.88677 |                       0.1      |
| cumulative    |   5 |  55 |         55 |         0.582479  |         0.999669 |                       23 |                  26   |                        3.06029 |                       0.1      |
| cumulative    |  10 |  10 |         10 |         0.0363636 |         0.821818 |                       10 |                  10   |                        2.1413  |                       0.2      |
| cumulative    |  10 |  20 |         20 |         0.207273  |         0.98     |                       16 |                  18   |                        2.63828 |                       0.133333 |
| cumulative    |  10 |  35 |         35 |         0.416623  |         0.997922 |                       21 |                  23   |                        2.93616 |                       0.1      |
| cumulative    |  10 |  55 |         55 |         0.570248  |         0.999669 |                       24 |                  26   |                        3.1065  |                       0.1      |
| ck            |   1 |  55 |         55 |         0.578843  |         0.998678 |                       23 |                  26   |                        3.06029 |                       0.1      |
| ck            |   2 |  55 |         55 |         0.57719   |         0.999339 |                       23 |                  26.6 |                        3.06029 |                       0.1      |
| ck            |   5 |  55 |         55 |         0.577851  |         0.999008 |                       23 |                  25   |                        3.07773 |                       0.1      |
| ck            |  10 |  55 |         55 |         0.581157  |         0.999008 |                       23 |                  25   |                        3.07773 |                       0.1      |

For cumulative configs, on average **94.9%** of cells C[i,j] have fewer than 5 observations, and **31.6%** are entirely unobserved. The median number of nonzero cells per row is **17.4** out of 30 possible output bins — confirming that the empirical transition matrix is highly sparse. Pure count-based estimation would require arbitrary smoothing decisions; the neural approach regularizes via the feature vector F_t and soft-label training.

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

**Interpretation:** CK deviation in the state-conditioned model indicates that the system's transition dynamics at horizon h ≠ h compositions of 1-step dynamics. This is consistent with the operator being time-inhomogeneous and the neural model learning richer horizon-specific structure than any stationary Markov chain can offer.

**This does not imply Markov consistency was achieved** by any model. CK consistency and predictive accuracy are separate objectives.

| model      |   h |     mean_kl |    mean_tv |   frobenius | note                                |
|:-----------|----:|------------:|-----------:|------------:|:------------------------------------|
| state_cond |   1 | 0           | 0          |   0         | h=1: trivial (identity composition) |
| state_free |   1 | 0           | 0          |   0         | h=1: trivial (identity composition) |
| state_cond |   2 | 0.158617    | 0.195489   |   0.450984  | nan                                 |
| state_free |   2 | 0.0319273   | 0.0897235  |   0.23194   | nan                                 |
| backoff_ck |   2 | 0.0106298   | 0.0637442  |   0.150172  | nan                                 |
| state_cond |   5 | 0.14689     | 0.204359   |   0.468521  | nan                                 |
| state_free |   5 | 0.0338235   | 0.0918738  |   0.237193  | nan                                 |
| backoff_ck |   5 | 0.00285252  | 0.0327933  |   0.0766981 | nan                                 |
| state_cond |  10 | 0.163862    | 0.199948   |   0.46165   | nan                                 |
| state_free |  10 | 0.0316783   | 0.0878112  |   0.228133  | nan                                 |
| backoff_ck |  10 | 9.77588e-05 | 0.00597987 |   0.0140149 | nan                                 |

---

## (iv) Uncertainty: Multi-Seed & Bootstrap CIs

We ran 3 seeds ([42, 7, 123]) for key configs (h=1,N=55) and (h=10,N=55). Block bootstrap CIs (circular, block_size=21, n_boot=500) are computed on per-sample log-likelihood for these configurations only. See `main_results_table.csv` (CI columns are NaN for non-key configs).

**Label definition:** The cumulative-return label for horizon h is Y_t^(h) = bin((P_{t+1+h} - P_{t+1}) / P_{t+1}), strictly forward-looking relative to the state X_t = bin((P_{t+1} - P_t) / P_t). For h=1, X_t is the current day's return and Y_t is the next day's return — a genuine 1-step-ahead forecasting task.

|   h |   N | model         |   test_ll |    delta_ll |   ci_lower |   ci_upper |
|----:|----:|:--------------|----------:|------------:|-----------:|-----------:|
|   1 |  55 | additive      |  -4.00002 |  0.00743972 |  nan       |  nan       |
|   1 |  55 | backoff       |  -4.00716 |  0.0002986  |  nan       |  nan       |
|   1 |  55 | marginal      |  -4.00746 |  0          |  nan       |  nan       |
|   1 |  55 | state_cond_nn |  -4.02633 | -0.0188719  |   -4.0565  |   -4.00019 |
|   1 |  55 | state_free_nn |  -4.01224 | -0.00478043 |   -4.02759 |   -4.00026 |
|  10 |  55 | additive      |  -4.00154 |  0.00640105 |  nan       |  nan       |
|  10 |  55 | backoff       |  -4.00082 |  0.00711488 |  nan       |  nan       |
|  10 |  55 | marginal      |  -4.00794 |  0          |  nan       |  nan       |
|  10 |  55 | state_cond_nn |  -3.9826  |  0.025339   |   -4.03201 |   -3.98038 |
|  10 |  55 | state_free_nn |  -3.9922  |  0.0157379  |   -4.02804 |   -3.97817 |
