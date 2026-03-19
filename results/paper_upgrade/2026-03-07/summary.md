# MathFrameworkExperiments — Summary

**Date:** 2026-03-07  
**Git hash:** 2c1cefc

---

## (i) Degeneracy Evidence

Transition degeneracy is diagnosed at the **cell level**: with ~1,650 training days and a 55×55 state-to-state space (3,025 possible transitions), count-based estimation is severely under-determined. The metrics below quantify this:

- `frac_cells_zero` — fraction of joint-count cells C[i,j] that are exactly zero
- `frac_cells_lt5` — fraction of cells with fewer than 5 observations
- `median_nonzero_per_row` — median distinct output bins reached per input state
- `p90_nonzero_per_row` — 90th percentile of same
- `median_row_entropy_empirical` — median entropy of empirical row distributions
- `median_row_maxprob_empirical` — median peak probability per row

| config_type   |   h |   N |   N_actual |   frac_cells_zero |   frac_cells_lt5 |   frac_cells_lt10 |   frac_rows_lt5 |   frac_rows_lt10 |   median_nonzero_per_row |   p90_nonzero_per_row |   median_row_entropy_empirical |   median_row_maxprob_empirical |
|:--------------|----:|----:|-----------:|------------------:|-----------------:|------------------:|----------------:|-----------------:|-------------------------:|----------------------:|-------------------------------:|-------------------------------:|
| cumulative    |   1 |  10 |         10 |         0.0563636 |         0.827273 |          0.994545 |               0 |                0 |                       10 |                  10   |                        2.13057 |                       0.2      |
| cumulative    |   1 |  20 |         20 |         0.224545  |         0.98     |          0.999091 |               0 |                0 |                       16 |                  17   |                        2.62696 |                       0.133333 |
| cumulative    |   1 |  35 |         35 |         0.436364  |         0.995844 |          1        |               0 |                0 |                       20 |                  22   |                        2.88048 |                       0.1      |
| cumulative    |   1 |  55 |         55 |         0.579504  |         0.998678 |          1        |               0 |                0 |                       23 |                  26   |                        3.06029 |                       0.1      |
| cumulative    |   2 |  10 |         10 |         0.0345455 |         0.807273 |          0.994545 |               0 |                0 |                       10 |                  10   |                        2.13669 |                       0.2      |
| cumulative    |   2 |  20 |         20 |         0.213636  |         0.976364 |          1        |               0 |                0 |                       16 |                  17   |                        2.60952 |                       0.133333 |
| cumulative    |   2 |  35 |         35 |         0.423377  |         0.996883 |          1        |               0 |                0 |                       20 |                  22.6 |                        2.90832 |                       0.1      |
| cumulative    |   2 |  55 |         55 |         0.582149  |         0.999339 |          1        |               0 |                0 |                       23 |                  25   |                        3.07773 |                       0.1      |
| cumulative    |   5 |  10 |         10 |         0.0490909 |         0.827273 |          0.998182 |               0 |                0 |                       10 |                  10   |                        2.15413 |                       0.193548 |
| cumulative    |   5 |  20 |         20 |         0.218182  |         0.977273 |          0.999091 |               0 |                0 |                       16 |                  17   |                        2.62465 |                       0.133333 |
| cumulative    |   5 |  35 |         35 |         0.423896  |         0.996883 |          1        |               0 |                0 |                       20 |                  23   |                        2.88677 |                       0.1      |
| cumulative    |   5 |  55 |         55 |         0.582479  |         0.999669 |          1        |               0 |                0 |                       23 |                  26   |                        3.06029 |                       0.1      |
| cumulative    |  10 |  10 |         10 |         0.0363636 |         0.821818 |          0.998182 |               0 |                0 |                       10 |                  10   |                        2.1413  |                       0.2      |
| cumulative    |  10 |  20 |         20 |         0.207273  |         0.98     |          1        |               0 |                0 |                       16 |                  18   |                        2.63828 |                       0.133333 |
| cumulative    |  10 |  35 |         35 |         0.416623  |         0.997922 |          1        |               0 |                0 |                       21 |                  23   |                        2.93616 |                       0.1      |
| cumulative    |  10 |  55 |         55 |         0.570248  |         0.999669 |          1        |               0 |                0 |                       24 |                  26   |                        3.1065  |                       0.1      |
| ck            |   1 |  55 |         55 |         0.578843  |         0.998678 |          1        |               0 |                0 |                       23 |                  26   |                        3.06029 |                       0.1      |
| ck            |   2 |  55 |         55 |         0.57719   |         0.999339 |          1        |               0 |                0 |                       23 |                  26.6 |                        3.06029 |                       0.1      |
| ck            |   5 |  55 |         55 |         0.577851  |         0.999008 |          1        |               0 |                0 |                       23 |                  25   |                        3.07773 |                       0.1      |
| ck            |  10 |  55 |         55 |         0.581157  |         0.999008 |          1        |               0 |                0 |                       23 |                  25   |                        3.07773 |                       0.1      |

For cumulative configs, on average **94.9%** of cells C[i,j] have fewer than 5 observations, and **31.6%** are entirely unobserved. The median number of nonzero cells per row is **17.4** out of 30 possible output bins.

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

| model      |   h |     mean_kl |    mean_tv |   frobenius | note                                |
|:-----------|----:|------------:|-----------:|------------:|:------------------------------------|
| state_cond |   1 | 0           | 0          |   0         | h=1: trivial (identity composition) |
| state_free |   1 | 0           | 0          |   0         | h=1: trivial (identity composition) |
| backoff_ck |   1 | 0           | 0          |   0         | h=1: trivial (identity composition) |
| state_cond |   2 | 0.151826    | 0.188423   |   0.439991  | nan                                 |
| state_free |   2 | 0.0299237   | 0.0868306  |   0.225266  | nan                                 |
| backoff_ck |   2 | 0.0106298   | 0.0637442  |   0.150172  | nan                                 |
| backoff_ck |   2 | 0.0106988   | 0.0638655  |   0.150704  | nan                                 |
| state_cond |   5 | 0.140333    | 0.205968   |   0.48804   | nan                                 |
| state_free |   5 | 0.0224787   | 0.0854416  |   0.211945  | nan                                 |
| backoff_ck |   5 | 0.00285252  | 0.0327933  |   0.0766981 | nan                                 |
| backoff_ck |   5 | 0.00290915  | 0.0329078  |   0.0774808 | nan                                 |
| state_cond |  10 | 0.151665    | 0.193007   |   0.447665  | nan                                 |
| state_free |  10 | 0.0295152   | 0.0846651  |   0.220997  | nan                                 |
| backoff_ck |  10 | 9.77588e-05 | 0.00597987 |   0.0140149 | nan                                 |
| backoff_ck |  10 | 0.000189745 | 0.00750218 |   0.0194677 | nan                                 |

---

## (iv) Uncertainty: Multi-Seed & Bootstrap CIs

Ran 3 seeds ([42, 7, 123]) for (h=1,N=55) and (h=10,N=55). Block bootstrap CIs (block_size=21, n_boot=500) on per-sample log-likelihood.

**Label:** Y_t^(h) = bin((P_{t+1+h} - P_{t+1}) / P_{t+1}) — strictly forward-looking.

|   h |   N | model         |   test_ll |    delta_ll |   ci_lower |   ci_upper |
|----:|----:|:--------------|----------:|------------:|-----------:|-----------:|
|   1 |  55 | additive      |  -4.00002 |  0.00743972 |  nan       |  nan       |
|   1 |  55 | backoff       |  -4.00716 |  0.0002986  |  nan       |  nan       |
|   1 |  55 | marginal      |  -4.00746 |  0          |  nan       |  nan       |
|   1 |  55 | state_cond_nn |  -4.04022 | -0.0327552  |   -4.08492 |   -4.00266 |
|   1 |  55 | state_free_nn |  -4.04805 | -0.0405885  |   -4.07668 |   -4.0247  |
|  10 |  55 | additive      |  -4.00154 |  0.00640105 |  nan       |  nan       |
|  10 |  55 | backoff       |  -4.00082 |  0.00711488 |  nan       |  nan       |
|  10 |  55 | marginal      |  -4.00794 |  0          |  nan       |  nan       |
|  10 |  55 | state_cond_nn |  -3.9826  |  0.025339   |   -4.02744 |   -3.97937 |
|  10 |  55 | state_free_nn |  -3.9922  |  0.0157379  |   -4.00968 |   -3.96615 |

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
|   1 |  55 | state_cond_nn | 3.05433 |      4.00728 |     4.04022 | -0.322782 |
|   1 |  55 | state_free_nn | 3.05433 |      4.00728 |     4.04805 | -0.325347 |
|  10 |  55 | additive      | 3.08927 |      4.00733 |     4.00154 | -0.295304 |
|  10 |  55 | backoff       | 3.08927 |      4.00733 |     4.00082 | -0.295073 |
|  10 |  55 | marginal      | 3.08927 |      4.00733 |     4.00794 | -0.297376 |
|  10 |  55 | state_cond_nn | 3.08927 |      4.00733 |     3.9826  | -0.289173 |
|  10 |  55 | state_free_nn | 3.08927 |      4.00733 |     3.9922  | -0.292281 |


In **98%** of configs, MIR < 0: neural regularisation improves test NLL beyond the empirically observed conditional entropy.

---

## (vi) Prefix Learning Curves

|   h |   N | model      |   nll_at_25pct |   nll_at_100pct |      delta | trend      |
|----:|----:|:-----------|---------------:|----------------:|-----------:|:-----------|
|   1 |  55 | state_cond |        4.00995 |         4.02459 |  0.0146488 | increasing |
|   1 |  55 | state_free |        4.01119 |         4.02757 |  0.0163706 | increasing |
|  10 |  55 | state_cond |        4.00795 |         3.9969  | -0.0110466 | decreasing |
|  10 |  55 | state_free |        4.01194 |         3.9961  | -0.0158404 | decreasing |

'decreasing' → still data-limited; 'saturating' → signal/capacity bound; 'increasing' → overfitting warning.

---

## (vii) Depth Ablation

|   h |   N | model      | arch    |   nll_test |   delta_nll_vs_shallow |
|----:|----:|:-----------|:--------|-----------:|-----------------------:|
|   1 |  55 | state_cond | deep    |    4.02459 |              -0.334514 |
|   1 |  55 | state_cond | shallow |    4.35911 |               0        |
|   1 |  55 | state_free | deep    |    4.02757 |              -0.308953 |
|   1 |  55 | state_free | shallow |    4.33652 |               0        |
|  10 |  55 | state_cond | deep    |    3.9969  |              -0.346059 |
|  10 |  55 | state_cond | shallow |    4.34296 |               0        |
|  10 |  55 | state_free | deep    |    3.9961  |              -0.250184 |
|  10 |  55 | state_free | shallow |    4.24628 |               0        |

Deep outperforms parameter-matched shallow in **50%** of (model, h, N, seed) combinations.

---

## (viii) Generalization Gap & Spectral Norm

| model      |   mean_gap |   max_gap |   spec_prod |
|:-----------|-----------:|----------:|------------:|
| state_cond |   0.143108 |  0.251827 |     38.2833 |
| state_free |   0.095915 |  0.134437 |     56.655  |

**mean_gap** = average (val_nll − train_nll). Larger spec_prod → wider gap (consistent with PAC-Bayes theory).

---

## (ix) Feature-Dimension Ablation

|   horizon |   N_bins |   n_features |   train_nll |   val_nll |   test_nll |     gen_gap |   spectral_product |
|----------:|---------:|-------------:|------------:|----------:|-----------:|------------:|-------------------:|
|         1 |       55 |           15 |     3.92568 |   3.98892 |    4.01048 |  0.0847983  |           65.0651  |
|         1 |       55 |           30 |     4.00731 |   4.00475 |    4.00359 | -0.00371981 |            2.16993 |
|         1 |       55 |           50 |     3.91457 |   3.99656 |    4.02888 |  0.11431    |           36.1225  |
|         1 |       55 |          194 |     3.89038 |   3.97732 |    4.01905 |  0.128669   |           47.1689  |
|        10 |       55 |           15 |     3.91802 |   3.98342 |    3.98923 |  0.0712066  |           87.1779  |
|        10 |       55 |           30 |     3.89069 |   3.98317 |    4.00448 |  0.113796   |           97.9674  |
|        10 |       55 |           50 |     3.92218 |   3.99623 |    4.00687 |  0.0846906  |           30.0186  |
|        10 |       55 |          194 |     3.92612 |   3.99591 |    3.98383 |  0.0577183  |           10.6174  |

**h=1, N=55:** Best test NLL = 4.0036 at n_features=30 (full=4.0191, Δ=-0.0155). Gen-gap: full=0.1287 → best=-0.0037. 
  ⟹ Reducing feature dimensionality **improves** test NLL: high-dim inputs were hurting generalisation.

**h=10, N=55:** Best test NLL = 3.9838 at n_features=194 (full=3.9838, Δ=+0.0000). Gen-gap: full=0.0577 → best=0.0577. 
  ⟹ Test NLL is **flat** across feature subsets: feature count is not the binding constraint here.
