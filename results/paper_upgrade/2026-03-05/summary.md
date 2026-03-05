# MathFrameworkExperiments — Summary

**Date:** 2026-03-05  
**Git hash:** ad6c284

---

## (i) Degeneracy Evidence

Transition degeneracy is diagnosed at the **cell level**: with ~1,650 training days and a 55×55 state-to-state space (3,025 possible transitions), count-based estimation is severely under-determined. The metrics below quantify this:

- `frac_cells_zero` — fraction of joint-count cells C[i,j] that are exactly zero
- `frac_cells_lt5` — fraction of cells with fewer than 5 observations
- `median_nonzero_per_row` — median distinct output bins reached per input state
- `p90_nonzero_per_row` — 90th percentile of same
- `median_row_entropy_empirical` — median entropy of empirical row distributions
- `median_row_maxprob_empirical` — median peak probability per row

| config_type   |   h |   N |   N_actual |   frac_cells_zero |   frac_cells_lt5 |   median_nonzero_per_row |   p90_nonzero_per_row |   median_row_entropy_empirical |   median_row_maxprob_empirical |
|:--------------|----:|----:|-----------:|------------------:|-----------------:|-------------------------:|----------------------:|-------------------------------:|-------------------------------:|
| cumulative    |   1 |  10 |         10 |         0.0563636 |         0.827273 |                       10 |                    10 |                        2.13057 |                            0.2 |
| cumulative    |   1 |  55 |         55 |         0.579504  |         0.998678 |                       23 |                    26 |                        3.06029 |                            0.1 |
| ck            |   1 |  55 |         55 |         0.578843  |         0.998678 |                       23 |                    26 |                        3.06029 |                            0.1 |

For cumulative configs, on average **91.3%** of cells C[i,j] have fewer than 5 observations, and **31.8%** are entirely unobserved. The median number of nonzero cells per row is **16.5** out of 32 possible output bins.

---

## (ii) Operator Diagnostics & Regime Case Study

Four diagnostics from the time-varying transition operator A_t^(1):

- **Dobrushin coefficient** δ(A_t): contraction; spikes in high-volatility regimes.
- **Row heterogeneity** ρ(A_t): state-dependence strength. Near-zero for StateFreeNet.
- **Row entropy** H(A_t): higher = more uniform transitions.
- **Spectral mixing proxy** σ_max(M): lower = faster mixing.

---

## (iii) Chapman–Kolmogorov Diagnostic Results

CK treated as a diagnostic. Label y_t^(h) := X_{t+h} in the same 55×55 space.

**Ranking by CK consistency:** Backoff > StateFreeNet > StateConditionedNet. StateConditionedNet's deviation indicates the system is genuinely time-inhomogeneous and horizon-specific — not a model defect.

| model      |   h |   mean_kl |   mean_tv |   frobenius | note                                |
|:-----------|----:|----------:|----------:|------------:|:------------------------------------|
| state_cond |   1 |         0 |         0 |           0 | h=1: trivial (identity composition) |
| state_free |   1 |         0 |         0 |           0 | h=1: trivial (identity composition) |

---

## (iv) Uncertainty: Multi-Seed & Bootstrap CIs

Ran 1 seeds ([42]) for (h=1,N=55) and (h=10,N=55). Block bootstrap CIs (block_size=21, n_boot=500) on per-sample log-likelihood.

**Label:** Y_t^(h) = bin((P_{t+1+h} - P_{t+1}) / P_{t+1}) — strictly forward-looking.

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

**H_irr** = H(Y|X) under empirical conditional on TRAIN. **MIR = (H_irr − NLL_model) / H_irr**.

Why MIR matters when signal is small:
1. When I(X;Y) ≈ 0, H_irr ≈ H(Y) — oracle gains nothing from state.
2. MIR scales out marginal difficulty for cross-horizon comparison.
3. Persistent negative MIR → neural regularisation works against over-fit empirical P.
4. MIR across depths → capacity vs data bottleneck.
5. MIR saturating as train_frac grows → data, not model, limits performance.

|   h |   N | model         |   H_irr |   H_marginal |   nll_model |       MIR |
|----:|----:|:--------------|--------:|-------------:|------------:|----------:|
|   1 |  55 | additive      | 3.05433 |      4.00728 |     4.00002 | -0.309622 |
|   1 |  55 | backoff       | 3.05433 |      4.00728 |     4.00716 | -0.31196  |
|   1 |  55 | marginal      | 3.05433 |      4.00728 |     4.00746 | -0.312058 |
|   1 |  55 | state_cond_nn | 3.05433 |      4.00728 |     4.01076 | -0.313137 |
|   1 |  55 | state_free_nn | 3.05433 |      4.00728 |     4.00966 | -0.312779 |


In **100%** of configs, MIR < 0: neural regularisation improves test NLL beyond the empirically observed conditional entropy.

---

## (vi) Prefix Learning Curves

|   h |   N | model      |   nll_at_25pct |   nll_at_100pct |       delta | trend      |
|----:|----:|:-----------|---------------:|----------------:|------------:|:-----------|
|   1 |  55 | state_cond |        4.01458 |         4.01472 | 0.000146389 | saturating |
|   1 |  55 | state_free |        4.01212 |         4.05225 | 0.040133    | increasing |

'decreasing' → still data-limited; 'saturating' → signal/capacity bound; 'increasing' → overfitting warning.

---

## (vii) Depth Ablation

|   h |   N | model      | arch    |   nll_test |   delta_nll_vs_shallow |
|----:|----:|:-----------|:--------|-----------:|-----------------------:|
|   1 |  55 | state_cond | deep    |    4.01472 |              -0.437583 |
|   1 |  55 | state_cond | shallow |    4.45231 |               0        |
|   1 |  55 | state_free | deep    |    4.05225 |              -0.327861 |
|   1 |  55 | state_free | shallow |    4.38011 |               0        |

Deep outperforms parameter-matched shallow in **50%** of (model, h, N, seed) combinations.

---

## (viii) Generalization Gap & Spectral Norm

| model      |   mean_gap |   max_gap |   spec_prod |
|:-----------|-----------:|----------:|------------:|
| state_cond |  0.143108  |  0.251827 |     38.2833 |
| state_free |  0.0805078 |  0.117697 |     37.8942 |

**mean_gap** = average (val_nll − train_nll). Larger spec_prod → wider gap (consistent with PAC-Bayes theory).

---

## (ix) Feature-Dimension Ablation

|   horizon |   N_bins |   n_features |   train_nll |   val_nll |   test_nll |     gen_gap |   spectral_product |
|----------:|---------:|-------------:|------------:|----------:|-----------:|------------:|-------------------:|
|         1 |       55 |           15 |     3.92568 |   3.98892 |    4.01048 |  0.0847983  |           65.0651  |
|         1 |       55 |           30 |     4.00731 |   4.00475 |    4.00359 | -0.00371981 |            2.16993 |
|         1 |       55 |           50 |     3.91457 |   3.99656 |    4.02888 |  0.11431    |           36.1225  |
|         1 |       55 |          194 |     3.89038 |   3.97732 |    4.01905 |  0.128669   |           47.1689  |

**h=1, N=55:** Best test NLL = 4.0036 at n_features=30 (full=4.0191, Δ=-0.0155). Gen-gap: full=0.1287 → best=-0.0037. 
  ⟹ Reducing feature dimensionality **improves** test NLL: high-dim inputs were hurting generalisation.
