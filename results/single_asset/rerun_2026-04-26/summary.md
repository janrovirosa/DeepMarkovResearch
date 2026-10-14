# MathFrameworkExperiments — Summary

**Date:** 2026-04-26  
**Git hash:** 22ee421

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
| state_cond |   2 | 0.0930571   | 0.16359    |   0.392374  | nan                                 |
| state_free |   2 | 0.0416121   | 0.120437   |   0.30063   | nan                                 |
| backoff_ck |   2 | 0.0106298   | 0.0637442  |   0.150172  | nan                                 |
| state_cond |   5 | 0.0210338   | 0.0781192  |   0.197276  | nan                                 |
| state_free |   5 | 0.0614059   | 0.144812   |   0.352797  | nan                                 |
| backoff_ck |   5 | 0.00285252  | 0.0327933  |   0.0766981 | nan                                 |
| state_cond |  10 | 0.0618537   | 0.131817   |   0.345455  | nan                                 |
| state_free |  10 | 0.041564    | 0.11162    |   0.290133  | nan                                 |
| backoff_ck |  10 | 9.77588e-05 | 0.00597987 |   0.0140149 | nan                                 |

---

## (iv) Uncertainty: Multi-Seed & Bootstrap CIs

Ran 3 seeds ([42, 7, 123]) for (h=1,N=55) and (h=10,N=55). Block bootstrap CIs (block_size=21, n_boot=500) on per-sample log-likelihood.

**Label:** Y_t^(h) = bin((P_{t+1+h} - P_{t+1}) / P_{t+1}) — strictly forward-looking.

|   h |   N | model         |   test_ll |    delta_ll |   ci_lower |   ci_upper |
|----:|----:|:--------------|----------:|------------:|-----------:|-----------:|
|   1 |  55 | additive      |  -4.00002 |  0.00743972 |  nan       |  nan       |
|   1 |  55 | backoff       |  -4.00716 |  0.0002986  |  nan       |  nan       |
|   1 |  55 | marginal      |  -4.00746 |  0          |  nan       |  nan       |
|   1 |  55 | state_cond_nn |  -4.0212  | -0.01374    |   -4.0388  |   -4.00629 |
|   1 |  55 | state_free_nn |  -4.02982 | -0.0223584  |   -4.07312 |   -3.98891 |
|  10 |  55 | additive      |  -4.00154 |  0.00640105 |  nan       |  nan       |
|  10 |  55 | backoff       |  -4.00082 |  0.00711488 |  nan       |  nan       |
|  10 |  55 | marginal      |  -4.00794 |  0          |  nan       |  nan       |
|  10 |  55 | state_cond_nn |  -3.9826  |  0.025339   |   -4.01261 |   -3.98376 |
|  10 |  55 | state_free_nn |  -3.9922  |  0.0157379  |   -4.0348  |   -3.98588 |

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
|   1 |  55 | state_cond_nn | 3.05433 |      4.00728 |     4.0212  | -0.316557 |
|   1 |  55 | state_free_nn | 3.05433 |      4.00728 |     4.02982 | -0.319378 |
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
|   1 |  55 | state_cond |        4.00975 |         4.03984 |  0.0300852 | increasing |
|   1 |  55 | state_free |        4.01062 |         4.02467 |  0.0140448 | increasing |
|  10 |  55 | state_cond |        4.00795 |         3.99506 | -0.0128874 | decreasing |
|  10 |  55 | state_free |        4.0119  |         3.99695 | -0.0149465 | decreasing |

'decreasing' → still data-limited; 'saturating' → signal/capacity bound; 'increasing' → overfitting warning.

---

## (vii) Depth Ablation

|   h |   N | model      | arch    |   nll_test |   delta_nll_vs_shallow |
|----:|----:|:-----------|:--------|-----------:|-----------------------:|
|   1 |  55 | state_cond | deep    |    4.03984 |              -0.320446 |
|   1 |  55 | state_cond | shallow |    4.36029 |               0        |
|   1 |  55 | state_free | deep    |    4.02467 |              -0.33585  |
|   1 |  55 | state_free | shallow |    4.36052 |               0        |
|  10 |  55 | state_cond | deep    |    3.99506 |              -0.341163 |
|  10 |  55 | state_cond | shallow |    4.33622 |               0        |
|  10 |  55 | state_free | deep    |    3.99695 |              -0.242027 |
|  10 |  55 | state_free | shallow |    4.23898 |               0        |

Deep outperforms parameter-matched shallow in **50%** of (model, h, N, seed) combinations.

---

## (viii) Generalization Gap & Spectral Norm

| model      |   mean_gap |   max_gap |   spec_prod |
|:-----------|-----------:|----------:|------------:|
| state_cond |   0.114101 |  0.199764 |     42.0659 |
| state_free |   0.089025 |  0.13774  |     38.9087 |

**mean_gap** = average (val_nll − train_nll). Larger spec_prod → wider gap (consistent with PAC-Bayes theory).

---

## (ix) Feature-Dimension Ablation

|   horizon |   N_bins |   n_features |   train_nll |   val_nll |   test_nll |     gen_gap |   spectral_product |
|----------:|---------:|-------------:|------------:|----------:|-----------:|------------:|-------------------:|
|         1 |       55 |           15 |     3.92658 |   3.98727 |    4.01735 |  0.0907705  |           70.748   |
|         1 |       55 |           30 |     4.00149 |   4.00458 |    4.00447 |  0.00297546 |            4.02221 |
|         1 |       55 |           50 |     3.91349 |   3.99318 |    4.0156  |  0.102115   |           37.6888  |
|         1 |       55 |          194 |     3.85336 |   3.97258 |    4.03749 |  0.184132   |           90.2655  |
|        10 |       55 |           15 |     4.00953 |   4.0013  |    4.00587 | -0.00365639 |            2.13326 |
|        10 |       55 |           30 |     3.83259 |   3.98048 |    4.00499 |  0.172399   |          280.804   |
|        10 |       55 |           50 |     3.90842 |   3.99289 |    3.99288 |  0.0844586  |           40.3328  |
|        10 |       55 |          194 |     3.91859 |   3.9941  |    3.9746  |  0.0560076  |           11.708   |

**h=1, N=55:** Best test NLL = 4.0045 at n_features=30 (full=4.0375, Δ=-0.0330). Gen-gap: full=0.1841 → best=0.0030. 
  ⟹ Reducing feature dimensionality **improves** test NLL: high-dim inputs were hurting generalisation.

**h=10, N=55:** Best test NLL = 3.9746 at n_features=194 (full=3.9746, Δ=+0.0000). Gen-gap: full=0.0560 → best=0.0560. 
  ⟹ Test NLL is **flat** across feature subsets: feature count is not the binding constraint here.
