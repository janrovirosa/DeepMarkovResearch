# Multi-Asset Experiment Summary

## Table 1 — Baseline NLL (state_cond vs state_free)

|                            |   nll_test |   delta_marginal |
|:---------------------------|-----------:|-----------------:|
| ('BAC', 'state_cond', 1)   |    3.99457 |     -0.0127667   |
| ('BAC', 'state_cond', 2)   |    4.00011 |     -0.00722685  |
| ('BAC', 'state_cond', 5)   |    4.00001 |     -0.00731983  |
| ('BAC', 'state_cond', 10)  |    4.01696 |      0.00962362  |
| ('BAC', 'state_free', 1)   |    3.99829 |     -0.00904622  |
| ('BAC', 'state_free', 2)   |    4.00024 |     -0.00709763  |
| ('BAC', 'state_free', 5)   |    4.02107 |      0.0137416   |
| ('BAC', 'state_free', 10)  |    4.00055 |     -0.00678196  |
| ('C', 'state_cond', 1)     |    3.95993 |     -0.0474037   |
| ('C', 'state_cond', 2)     |    3.97066 |     -0.0366689   |
| ('C', 'state_cond', 5)     |    3.97045 |     -0.0368845   |
| ('C', 'state_cond', 10)    |    3.99187 |     -0.0154592   |
| ('C', 'state_free', 1)     |    3.9504  |     -0.0569369   |
| ('C', 'state_free', 2)     |    3.9753  |     -0.0320376   |
| ('C', 'state_free', 5)     |    3.97837 |     -0.0289642   |
| ('C', 'state_free', 10)    |    3.99073 |     -0.0166039   |
| ('FITB', 'state_cond', 1)  |    3.96101 |     -0.0463239   |
| ('FITB', 'state_cond', 2)  |    3.98897 |     -0.0183631   |
| ('FITB', 'state_cond', 5)  |    3.98527 |     -0.0220603   |
| ('FITB', 'state_cond', 10) |    4.01279 |      0.00545606  |
| ('FITB', 'state_free', 1)  |    3.96693 |     -0.0403985   |
| ('FITB', 'state_free', 2)  |    3.97973 |     -0.0275985   |
| ('FITB', 'state_free', 5)  |    4.00157 |     -0.00576344  |
| ('FITB', 'state_free', 10) |    4.01743 |      0.0101014   |
| ('GS', 'state_cond', 1)    |    4.0035  |     -0.00382939  |
| ('GS', 'state_cond', 2)    |    4.02743 |      0.0200959   |
| ('GS', 'state_cond', 5)    |    4.03087 |      0.0235392   |
| ('GS', 'state_cond', 10)   |    4.00727 |     -6.66638e-05 |
| ('GS', 'state_free', 1)    |    4.00543 |     -0.00190296  |
| ('GS', 'state_free', 2)    |    4.03439 |      0.0270587   |
| ('GS', 'state_free', 5)    |    4.0153  |      0.00796661  |
| ('GS', 'state_free', 10)   |    4.00644 |     -0.00089493  |
| ('JPM', 'state_cond', 1)   |    3.95528 |     -0.0520562   |
| ('JPM', 'state_cond', 2)   |    3.96928 |     -0.0380572   |
| ('JPM', 'state_cond', 5)   |    3.97067 |     -0.0366584   |
| ('JPM', 'state_cond', 10)  |    3.9775  |     -0.0298284   |
| ('JPM', 'state_free', 1)   |    3.95365 |     -0.0536822   |
| ('JPM', 'state_free', 2)   |    3.97479 |     -0.0325459   |
| ('JPM', 'state_free', 5)   |    3.97005 |     -0.0372857   |
| ('JPM', 'state_free', 10)  |    3.98294 |     -0.024393    |
| ('MS', 'state_cond', 1)    |    3.96878 |     -0.0385555   |
| ('MS', 'state_cond', 2)    |    3.99028 |     -0.0170547   |
| ('MS', 'state_cond', 5)    |    4.00195 |     -0.00537958  |
| ('MS', 'state_cond', 10)   |    4.0015  |     -0.00583401  |
| ('MS', 'state_free', 1)    |    3.96747 |     -0.0398673   |
| ('MS', 'state_free', 2)    |    4.00169 |     -0.00564041  |
| ('MS', 'state_free', 5)    |    3.99877 |     -0.00856128  |
| ('MS', 'state_free', 10)   |    4.0037  |     -0.00363293  |
| ('MTB', 'state_cond', 1)   |    3.95011 |     -0.0572211   |
| ('MTB', 'state_cond', 2)   |    3.99112 |     -0.0162124   |
| ('MTB', 'state_cond', 5)   |    3.98968 |     -0.017655    |
| ('MTB', 'state_cond', 10)  |    3.97646 |     -0.0308682   |
| ('MTB', 'state_free', 1)   |    3.94059 |     -0.0667447   |
| ('MTB', 'state_free', 2)   |    4.01824 |      0.0109111   |
| ('MTB', 'state_free', 5)   |    3.99982 |     -0.00751414  |
| ('MTB', 'state_free', 10)  |    3.99007 |     -0.0172667   |
| ('PNC', 'state_cond', 1)   |    3.98145 |     -0.0258864   |
| ('PNC', 'state_cond', 2)   |    4.00657 |     -0.000765707 |
| ('PNC', 'state_cond', 5)   |    4.01942 |      0.0120827   |
| ('PNC', 'state_cond', 10)  |    3.99516 |     -0.0121702   |
| ('PNC', 'state_free', 1)   |    3.99816 |     -0.00917163  |
| ('PNC', 'state_free', 2)   |    3.99917 |     -0.00816622  |
| ('PNC', 'state_free', 5)   |    3.99741 |     -0.00991883  |
| ('PNC', 'state_free', 10)  |    3.9981  |     -0.00923338  |
| ('USB', 'state_cond', 1)   |    4.05666 |      0.0493308   |
| ('USB', 'state_cond', 2)   |    4.07042 |      0.0630856   |
| ('USB', 'state_cond', 5)   |    4.04256 |      0.0352221   |
| ('USB', 'state_cond', 10)  |    4.09765 |      0.0903131   |
| ('USB', 'state_free', 1)   |    4.09911 |      0.0917784   |
| ('USB', 'state_free', 2)   |    4.0528  |      0.0454708   |
| ('USB', 'state_free', 5)   |    4.07336 |      0.0660263   |
| ('USB', 'state_free', 10)  |    4.11316 |      0.105827    |
| ('WFC', 'state_cond', 1)   |    3.97508 |     -0.0322486   |
| ('WFC', 'state_cond', 2)   |    3.98455 |     -0.0227837   |
| ('WFC', 'state_cond', 5)   |    3.99239 |     -0.0149449   |
| ('WFC', 'state_cond', 10)  |    3.97876 |     -0.0285777   |
| ('WFC', 'state_free', 1)   |    3.99408 |     -0.0132557   |
| ('WFC', 'state_free', 2)   |    3.99564 |     -0.0116912   |
| ('WFC', 'state_free', 5)   |    3.99182 |     -0.0155167   |
| ('WFC', 'state_free', 10)  |    3.98754 |     -0.0197965   |


## Table 2 — Sigma Sweep on JPM (h=1, N=55)

|   sigma |   nll_val |   nll_test |
|--------:|----------:|-----------:|
|     0.5 |   3.83998 |    3.95826 |
|     1   |   3.83531 |    3.94803 |
|     1.5 |   3.83541 |    3.94654 |
|     2   |   3.83614 |    3.94564 |
|     3   |   3.83707 |    3.94909 |


## Table 3 — SWA + best_sigma=1.0 Improvement over Baseline

> **Methodological note**: `BEST_SIGMA` (1.0) was selected via sweep on JPM only (Experiment B) and applied uniformly to all 10 tickers in Experiment C. This is intentional — per-ticker sigma selection on validation data would risk overfitting the bandwidth to individual stocks.

|                        |       delta |
|:-----------------------|------------:|
| ('BAC', 'state_cond')  | 0.0109736   |
| ('BAC', 'state_free')  | 0.0140625   |
| ('C', 'state_cond')    | 0.00787675  |
| ('C', 'state_free')    | 0.00413209  |
| ('FITB', 'state_cond') | 0.0115625   |
| ('FITB', 'state_free') | 0.00783408  |
| ('GS', 'state_cond')   | 0.0189806   |
| ('GS', 'state_free')   | 0.0105104   |
| ('JPM', 'state_cond')  | 0.00104745  |
| ('JPM', 'state_free')  | 0.000592629 |
| ('MS', 'state_cond')   | 0.0124483   |
| ('MS', 'state_free')   | 0.00106555  |
| ('MTB', 'state_cond')  | 0.0171153   |
| ('MTB', 'state_free')  | 0.0135773   |
| ('PNC', 'state_cond')  | 0.015447    |
| ('PNC', 'state_free')  | 0.0148814   |
| ('USB', 'state_cond')  | 0.0525347   |
| ('USB', 'state_free')  | 0.0303437   |
| ('WFC', 'state_cond')  | 0.0162177   |
| ('WFC', 'state_free')  | 0.0144379   |


## Table 4 — Higher-order k Ablation (JPM, h=1)

| model   |   nll_val |   nll_test |   delta_marginal |
|:--------|----------:|-----------:|-----------------:|
| ho_k1   |   3.84123 |    3.95174 |       -0.055591  |
| ho_k2   |   3.84748 |    3.9574  |       -0.0499302 |
| ho_k3   |   3.84415 |    3.95665 |       -0.0506826 |
| ho_k5   |   3.85205 |    3.96763 |       -0.0397014 |


## Table 5 — Conditioning Regime Comparison (JPM, h=1)

| model                 |   nll_val |   nll_test |   delta_marginal |
|:----------------------|----------:|-----------:|-----------------:|
| state_cond_full       |   3.84123 |    3.95174 |       -0.055591  |
| state_cond_macro_only |   3.83782 |    3.94767 |       -0.0596645 |
| state_cond_own_only   |   3.85513 |    3.96294 |       -0.0443895 |


## Summary

- **Baseline**: State-conditioned models trained on all 10 bank stocks with sigma=1.0.
- **Sigma sweep**: Performed on JPM only; best sigma = 1.0.
- **SWA**: Rolling average of last 10 checkpoints applied with best sigma.
- **Higher-order Markov**: k-step history ablation on JPM.
- **Regime conditioning**: Compares full features vs macro-only vs own-stock lags.


---
# Extended Experiment Summary (A-EXT through F-NEW)

## Per-Stock Best Configuration (argmin delta_marginal over all h,N,regime)

| ticker   | exp_tag                 | model                 |   h |   N |   delta_marginal |
|:---------|:------------------------|:----------------------|----:|----:|-----------------:|
| JPM      | swa_bestsigma_ext       | state_free            |   1 |  55 |      -0.0629438  |
| C        | conditioning_regime_ext | state_cond_macro_only |   1 |  55 |      -0.0501472  |
| WFC      | conditioning_regime_ext | state_cond_own_only   |  21 |  55 |      -0.0604424  |
| GS       | conditioning_regime_ext | state_cond_own_only   |   1 |  55 |      -0.00642958 |
| MS       | conditioning_regime_ext | state_cond_own_only   |   1 |  55 |      -0.0499562  |
| PNC      | conditioning_regime_ext | state_cond_macro_only |   1 |  55 |      -0.0412661  |
| USB      | conditioning_regime_ext | state_cond_own_only   |   1 |  55 |       0.00123129 |
| FITB     | conditioning_regime_ext | state_cond_macro_only |   1 |  55 |      -0.0623607  |
| MTB      | swa_bestsigma_ext       | state_cond            |   1 |  35 |      -0.0771128  |
| BAC      | swa_bestsigma_ext       | state_free            |   1 |  35 |      -0.0199629  |


## Pooled delta_marginal (meanÂ±std across 10 stocks)

### By horizon h

|   h |        mean |       std |
|----:|------------:|----------:|
|   1 | -0.0107944  | 0.0362508 |
|   5 |  0.00526823 | 0.0281409 |
|  10 |  0.0256331  | 0.0434953 |
|  21 |  0.038354   | 0.0651972 |
|  63 |  0.112775   | 0.129024  |

### By N bins

|   N |      mean |       std |
|----:|----------:|----------:|
|   4 | 0.0318207 | 0.0347214 |
|  10 | 0.0230961 | 0.0605371 |
|  20 | 0.0367441 | 0.092036  |
|  35 | 0.0394219 | 0.101579  |
|  55 | 0.0401536 | 0.100967  |


## Cross-Asset Synchrony

Mean off-diagonal Pearson correlation of row entropy: **0.6462**

## Failure Diagnostic (delta_marginal > 0)

| ticker   | exp_tag                 | model                       |   h |   N |   seed |   delta_marginal |
|:---------|:------------------------|:----------------------------|----:|----:|-------:|-----------------:|
| JPM      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.00336385  |
| JPM      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.000642184 |
| JPM      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.00376726  |
| JPM      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.00298047  |
| JPM      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.00206638  |
| JPM      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.00246144  |
| JPM      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.00123096  |
| JPM      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.0066655   |
| JPM      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.00426057  |
| JPM      | baseline_ext            | state_cond                  |  63 |  10 |    123 |      0.00197795  |
| JPM      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.00927547  |
| JPM      | baseline_ext            | state_free                  |  63 |  10 |      7 |      0.00215677  |
| JPM      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.0111175   |
| JPM      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.00113586  |
| JPM      | baseline_ext            | state_free                  |  21 |  20 |     42 |      0.00147155  |
| JPM      | baseline_ext            | state_cond                  |  63 |  20 |     42 |      0.0519398   |
| JPM      | baseline_ext            | state_cond                  |  63 |  20 |      7 |      0.00292734  |
| JPM      | baseline_ext            | state_cond                  |  63 |  20 |    123 |      0.0145142   |
| JPM      | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.0164366   |
| JPM      | baseline_ext            | state_free                  |  63 |  20 |      7 |      0.00445369  |
| JPM      | baseline_ext            | state_free                  |  63 |  20 |    123 |      0.00742367  |
| JPM      | baseline_ext            | state_cond                  |  63 |  35 |     42 |      0.0315388   |
| JPM      | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.013631    |
| JPM      | baseline_ext            | state_cond                  |  63 |  35 |    123 |      0.0412673   |
| JPM      | baseline_ext            | state_free                  |  63 |  35 |      7 |      0.0197147   |
| JPM      | baseline_ext            | state_free                  |  63 |  35 |    123 |      0.0163041   |
| JPM      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.0173007   |
| JPM      | baseline_ext            | state_cond                  |  63 |  55 |      7 |      0.0221506   |
| JPM      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.00335178  |
| JPM      | baseline_ext            | state_free                  |  63 |  55 |     42 |      0.0167814   |
| JPM      | baseline_ext            | state_free                  |  63 |  55 |      7 |      0.0233737   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.000229005 |
| JPM      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.00168848  |
| JPM      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0021329   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0649427   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0520668   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0143584   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.039829    |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0211581   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0315036   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0179041   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.0292421   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.024921    |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.00682429  |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.0205732   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.00722268  |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.0813902   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.057283    |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.0723799   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.0873533   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.0397003   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.0319395   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.0850233   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.0815767   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.0618883   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.0852486   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0622848   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.048297    |
| JPM      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.00642166  |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.0962492   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.0864793   |
| JPM      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.0351282   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.0868598   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0611387   |
| JPM      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.0072175   |
| JPM      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.00642166  |
| JPM      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0197035   |
| JPM      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0350123   |
| JPM      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.00477991  |
| JPM      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0395213   |
| JPM      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0311323   |
| JPM      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0287572   |
| C        | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0102792   |
| C        | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.00117278  |
| C        | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.00357843  |
| C        | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0100515   |
| C        | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.00298357  |
| C        | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.00482059  |
| C        | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0166187   |
| C        | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0123519   |
| C        | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0129679   |
| C        | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0161971   |
| C        | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.014475    |
| C        | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0104047   |
| C        | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.0273924   |
| C        | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.023585    |
| C        | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0193257   |
| C        | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0310947   |
| C        | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0266131   |
| C        | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0256442   |
| C        | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0416707   |
| C        | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.026651    |
| C        | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.0620184   |
| C        | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0467098   |
| C        | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0295218   |
| C        | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0610269   |
| C        | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.0258969   |
| C        | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0438563   |
| C        | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.0556488   |
| C        | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.0474678   |
| C        | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.0748978   |
| C        | baseline_ext            | state_free                  |  63 |   4 |    123 |      0.0884556   |
| C        | baseline_ext            | state_cond                  |   5 |  10 |    123 |      0.0023835   |
| C        | baseline_ext            | state_cond                  |  10 |  10 |    123 |      0.0347145   |
| C        | baseline_ext            | state_free                  |  10 |  10 |     42 |      0.000859054 |
| C        | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.000769647 |
| C        | baseline_ext            | state_cond                  |  21 |  10 |    123 |      0.0312727   |
| C        | baseline_ext            | state_free                  |  21 |  10 |     42 |      0.0371905   |
| C        | baseline_ext            | state_free                  |  21 |  10 |      7 |      0.0303414   |
| C        | baseline_ext            | state_free                  |  21 |  10 |    123 |      0.0333801   |
| C        | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.00142125  |
| C        | baseline_ext            | state_cond                  |  10 |  20 |    123 |      0.00812367  |
| C        | baseline_ext            | state_free                  |  10 |  20 |     42 |      0.0133701   |
| C        | baseline_ext            | state_free                  |  10 |  20 |      7 |      2.57831e-05 |
| C        | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.00666526  |
| C        | baseline_ext            | state_free                  |  21 |  20 |     42 |      0.00309471  |
| C        | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.00628427  |
| C        | baseline_ext            | state_cond                  |  63 |  20 |     42 |      0.0155738   |
| C        | baseline_ext            | state_cond                  |  63 |  20 |      7 |      0.00360588  |
| C        | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.00411585  |
| C        | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.0104748   |
| C        | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.00385819  |
| C        | baseline_ext            | state_cond                  |  21 |  35 |     42 |      0.00289021  |
| C        | baseline_ext            | state_cond                  |  21 |  35 |      7 |      0.00512705  |
| C        | baseline_ext            | state_cond                  |  21 |  35 |    123 |      0.00902519  |
| C        | baseline_ext            | state_free                  |  21 |  35 |     42 |      0.0207373   |
| C        | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.00244269  |
| C        | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.0105804   |
| C        | baseline_ext            | state_free                  |  63 |  35 |      7 |      0.00105915  |
| C        | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.00964126  |
| C        | baseline_ext            | state_cond                  |  21 |  55 |     42 |      0.00985822  |
| C        | baseline_ext            | state_cond                  |  21 |  55 |      7 |      0.0173489   |
| C        | baseline_ext            | state_cond                  |  21 |  55 |    123 |      0.00496158  |
| C        | baseline_ext            | state_free                  |  21 |  55 |     42 |      0.0182844   |
| C        | baseline_ext            | state_free                  |  21 |  55 |      7 |      0.00384626  |
| C        | baseline_ext            | state_cond                  |  63 |  55 |      7 |      0.00826797  |
| C        | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.0110126   |
| C        | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.00298322  |
| C        | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0050658   |
| C        | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.00573922  |
| C        | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.00570548  |
| C        | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.00297916  |
| C        | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.00497556  |
| C        | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.00698245  |
| C        | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.00588429  |
| C        | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.00846577  |
| C        | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.00703168  |
| C        | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.00845349  |
| C        | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0103054   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0406269   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0384239   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0279809   |
| C        | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0257618   |
| C        | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0324889   |
| C        | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0253296   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0835291   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0660867   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.109172    |
| C        | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0938337   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.05663     |
| C        | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0860785   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.146735    |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.219188    |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.106173    |
| C        | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.112782    |
| C        | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.159947    |
| C        | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.247045    |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.0043042   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.00924805  |
| C        | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      8.75316e-05 |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.0313707   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.0415757   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.0857137   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.0500837   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.0773127   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.00600556  |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0157395   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.00121311  |
| C        | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.00257424  |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.0303514   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.0152216   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.0269599   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.0136159   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.000156675 |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.0391803   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.0432594   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.0221463   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.0303462   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.0468274   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.0216942   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.0156253   |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.00172696  |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.00180898  |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.0312539   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.0312747   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.0468958   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.0359827   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.00695739  |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.00118694  |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.0449756   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.041102    |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.019203    |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.0591756   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0306867   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.029467    |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.00314626  |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.00804577  |
| C        | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.00543556  |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.0331694   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.0704809   |
| C        | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.0520917   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0378529   |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.00885353  |
| C        | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.014947    |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.0521422   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.0397469   |
| C        | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.0362416   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.0516625   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0225998   |
| C        | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.0130073   |
| C        | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.0331694   |
| C        | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.0704809   |
| C        | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.0520917   |
| C        | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0667454   |
| C        | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0616904   |
| C        | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.051869    |
| C        | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.046568    |
| C        | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.0403782   |
| C        | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.0478931   |
| C        | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0245558   |
| C        | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0242563   |
| C        | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0164591   |
| WFC      | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0156816   |
| WFC      | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.0252938   |
| WFC      | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.0230088   |
| WFC      | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0215157   |
| WFC      | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.0203071   |
| WFC      | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0247339   |
| WFC      | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.053788    |
| WFC      | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0472869   |
| WFC      | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0587783   |
| WFC      | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.062704    |
| WFC      | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0583133   |
| WFC      | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0542176   |
| WFC      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.0624073   |
| WFC      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.07319     |
| WFC      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0737319   |
| WFC      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0758492   |
| WFC      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0473179   |
| WFC      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0655422   |
| WFC      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0522157   |
| WFC      | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0528119   |
| WFC      | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.0491698   |
| WFC      | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0401424   |
| WFC      | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0506434   |
| WFC      | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0545149   |
| WFC      | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.0357603   |
| WFC      | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0302011   |
| WFC      | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.0166168   |
| WFC      | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.0483643   |
| WFC      | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.0343654   |
| WFC      | baseline_ext            | state_free                  |  63 |   4 |    123 |      0.0411578   |
| WFC      | baseline_ext            | state_cond                  |   5 |  10 |      7 |      0.00401381  |
| WFC      | baseline_ext            | state_cond                  |  10 |  10 |      7 |      0.00854662  |
| WFC      | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.0168996   |
| WFC      | baseline_ext            | state_cond                  |  21 |  10 |    123 |      0.00571683  |
| WFC      | baseline_ext            | state_free                  |  21 |  10 |     42 |      0.00165847  |
| WFC      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.0108655   |
| WFC      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.0100236   |
| WFC      | baseline_ext            | state_cond                  |  63 |  10 |    123 |      0.005492    |
| WFC      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.00605467  |
| WFC      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.000943454 |
| WFC      | baseline_ext            | state_cond                  |   5 |  20 |     42 |      0.000857149 |
| WFC      | baseline_ext            | state_free                  |   5 |  20 |    123 |      0.0026367   |
| WFC      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.000850473 |
| WFC      | baseline_ext            | state_free                  |   1 |  35 |      7 |      0.00631414  |
| WFC      | baseline_ext            | state_cond                  |   5 |  35 |     42 |      0.00992761  |
| WFC      | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.0106021   |
| WFC      | baseline_ext            | state_free                  |  63 |  35 |     42 |      0.00583038  |
| WFC      | baseline_ext            | state_cond                  |  21 |  55 |     42 |      0.00786695  |
| WFC      | baseline_ext            | state_cond                  |  21 |  55 |    123 |      0.0117846   |
| WFC      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.00906954  |
| WFC      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.0083271   |
| WFC      | baseline_ext            | state_free                  |  63 |  55 |      7 |      0.00251875  |
| WFC      | baseline_ext            | state_free                  |  63 |  55 |    123 |      0.00496969  |
| WFC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.0182939   |
| WFC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0221583   |
| WFC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.0224792   |
| WFC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.0202328   |
| WFC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.0217149   |
| WFC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.023597    |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.041458    |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.0466815   |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0417033   |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0398476   |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.0415447   |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0394294   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0488385   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0510972   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0476601   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.049749    |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0557375   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0529199   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0498525   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0588172   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0545729   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0646931   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0529681   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0558912   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0494248   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0549436   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0668939   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.107317    |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0390253   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0977284   |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |      7 |      0.00401166  |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  10 |     42 |      0.00361446  |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  10 |      7 |      0.00240353  |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  10 |    123 |      0.00421956  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.00438956  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.00103143  |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.00841502  |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.00496915  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.0232597   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.00194314  |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.0124536   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.0117414   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.0184579   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0505112   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.0439775   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.039619    |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.0512839   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.0574746   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.0372074   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0164712   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.00703624  |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.000844751 |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.0180605   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.0146935   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.0091348   |
| WFC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.00900534  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.0587819   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.0810476   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.0437093   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.0658589   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.0598026   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.0813654   |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |      7 |      0.00110231  |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.00556741  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0140859   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0133544   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.00646625  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.0358175   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.05332     |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.0455779   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.0615891   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0612446   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.0517677   |
| WFC      | swa_bestsigma_ext       | state_free                  |   1 |  55 |      7 |      0.00443277  |
| WFC      | swa_bestsigma_ext       | state_free                  |   1 |  55 |    123 |      0.011171    |
| WFC      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |      7 |      0.0112658   |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.000609491 |
| WFC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.00114546  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.00161562  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0212799   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.00911102  |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.00948438  |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.0159942   |
| WFC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.0144988   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.00487003  |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.0253421   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.0452004   |
| WFC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.0459453   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.0435997   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0363413   |
| WFC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.0466972   |
| WFC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.00487003  |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |     42 |      0.00653038  |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |      7 |      0.0224472   |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |    123 |      0.0010725   |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.091775    |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.111764    |
| WFC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.0317036   |
| WFC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |     42 |      0.0203186   |
| WFC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |      7 |      0.0321613   |
| WFC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |    123 |      0.0254036   |
| GS       | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.000815872 |
| GS       | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.00147415  |
| GS       | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.00119913  |
| GS       | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.000710491 |
| GS       | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0089562   |
| GS       | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0108162   |
| GS       | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0138379   |
| GS       | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.00953794  |
| GS       | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.013795    |
| GS       | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0108681   |
| GS       | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.0200503   |
| GS       | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.0184283   |
| GS       | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0248432   |
| GS       | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0163492   |
| GS       | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0228207   |
| GS       | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0208789   |
| GS       | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.00573981  |
| GS       | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.00229264  |
| GS       | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.00199616  |
| GS       | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.00847507  |
| GS       | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.00166214  |
| GS       | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0061816   |
| GS       | baseline_ext            | state_cond                  |   1 |  10 |      7 |      0.00185994  |
| GS       | baseline_ext            | state_cond                  |   5 |  10 |     42 |      0.00851754  |
| GS       | baseline_ext            | state_cond                  |   5 |  10 |      7 |      0.00962118  |
| GS       | baseline_ext            | state_cond                  |   5 |  10 |    123 |      0.00973872  |
| GS       | baseline_ext            | state_free                  |   5 |  10 |     42 |      0.00712541  |
| GS       | baseline_ext            | state_free                  |   5 |  10 |      7 |      0.0126493   |
| GS       | baseline_ext            | state_free                  |   5 |  10 |    123 |      0.0130041   |
| GS       | baseline_ext            | state_cond                  |  21 |  10 |     42 |      0.00614408  |
| GS       | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.00326041  |
| GS       | baseline_ext            | state_free                  |  21 |  10 |      7 |      0.004329    |
| GS       | baseline_ext            | state_cond                  |   1 |  20 |     42 |      0.00235227  |
| GS       | baseline_ext            | state_free                  |   1 |  20 |      7 |      0.00235299  |
| GS       | baseline_ext            | state_cond                  |   5 |  20 |     42 |      0.00438503  |
| GS       | baseline_ext            | state_cond                  |   5 |  20 |      7 |      0.0120309   |
| GS       | baseline_ext            | state_cond                  |   5 |  20 |    123 |      0.0131002   |
| GS       | baseline_ext            | state_free                  |   5 |  20 |     42 |      0.0236929   |
| GS       | baseline_ext            | state_free                  |   5 |  20 |      7 |      0.0120166   |
| GS       | baseline_ext            | state_free                  |   5 |  20 |    123 |      0.0120027   |
| GS       | baseline_ext            | state_cond                  |  10 |  20 |      7 |      0.00202874  |
| GS       | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.000270162 |
| GS       | baseline_ext            | state_cond                  |  21 |  20 |      7 |      0.000586544 |
| GS       | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.00297955  |
| GS       | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.0220604   |
| GS       | baseline_ext            | state_cond                  |   1 |  35 |     42 |      0.00698958  |
| GS       | baseline_ext            | state_cond                  |   1 |  35 |      7 |      0.016666    |
| GS       | baseline_ext            | state_free                  |   1 |  35 |     42 |      0.00908051  |
| GS       | baseline_ext            | state_free                  |   1 |  35 |      7 |      0.00692163  |
| GS       | baseline_ext            | state_free                  |   1 |  35 |    123 |      0.00907312  |
| GS       | baseline_ext            | state_cond                  |   5 |  35 |     42 |      0.0108806   |
| GS       | baseline_ext            | state_cond                  |   5 |  35 |      7 |      0.0106438   |
| GS       | baseline_ext            | state_cond                  |   5 |  35 |    123 |      0.0103463   |
| GS       | baseline_ext            | state_free                  |   5 |  35 |     42 |      0.0113417   |
| GS       | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.0219222   |
| GS       | baseline_ext            | state_free                  |   5 |  35 |    123 |      0.0185641   |
| GS       | baseline_ext            | state_free                  |  10 |  35 |     42 |      0.00506697  |
| GS       | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.00154839  |
| GS       | baseline_ext            | state_free                  |  10 |  35 |    123 |      0.00411759  |
| GS       | baseline_ext            | state_free                  |  21 |  35 |     42 |      0.0246323   |
| GS       | baseline_ext            | state_cond                  |   1 |  55 |      7 |      0.00248823  |
| GS       | baseline_ext            | state_free                  |   1 |  55 |      7 |      0.0133406   |
| GS       | baseline_ext            | state_free                  |   1 |  55 |    123 |      0.010027    |
| GS       | baseline_ext            | state_cond                  |   5 |  55 |     42 |      0.0235392   |
| GS       | baseline_ext            | state_cond                  |   5 |  55 |      7 |      0.0155521   |
| GS       | baseline_ext            | state_cond                  |   5 |  55 |    123 |      0.00530014  |
| GS       | baseline_ext            | state_free                  |   5 |  55 |     42 |      0.00796661  |
| GS       | baseline_ext            | state_free                  |   5 |  55 |      7 |      0.0210105   |
| GS       | baseline_ext            | state_free                  |   5 |  55 |    123 |      0.00802622  |
| GS       | baseline_ext            | state_cond                  |  10 |  55 |    123 |      0.00387726  |
| GS       | baseline_ext            | state_free                  |  10 |  55 |      7 |      0.00204334  |
| GS       | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.0011383   |
| GS       | baseline_ext            | state_cond                  |  21 |  55 |      7 |      0.00202045  |
| GS       | baseline_ext            | state_cond                  |  21 |  55 |    123 |      0.0111838   |
| GS       | baseline_ext            | state_free                  |  21 |  55 |    123 |      0.00257215  |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.00207615  |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.000827912 |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.00185466  |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.00152612  |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.0089947   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.00998855  |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0106971   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.00890232  |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.00790513  |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.00761676  |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0194746   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.015285    |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0162802   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0222588   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0186877   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0181702   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0321968   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.037164    |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0308048   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0300064   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0284785   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0260519   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0189422   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.00999606  |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.00375832  |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0278522   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.00466824  |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0107725   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  10 |     42 |      0.00443724  |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  10 |      7 |      0.00625685  |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  10 |    123 |      0.00366453  |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  10 |     42 |      0.0085893   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  10 |      7 |      0.0131154   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  10 |    123 |      0.0213774   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  10 |     42 |      0.0197707   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  10 |      7 |      0.0173953   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  10 |    123 |      0.0236647   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.00448731  |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.00542096  |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.0146623   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.00527171  |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.0055621   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.0103801   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  10 |     42 |      0.0230494   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.0603631   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.0407188   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.030067    |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.0791698   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.0191532   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0894292   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.0872982   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.0788012   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.064725    |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.0636289   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.0813811   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  20 |     42 |      0.0227409   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  20 |      7 |      0.0137077   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  20 |    123 |      0.00934104  |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  20 |      7 |      0.0078483   |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  20 |    123 |      0.00408915  |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  20 |     42 |      0.0238133   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  20 |      7 |      0.0298643   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  20 |    123 |      0.0193849   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  20 |     42 |      0.0358971   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  20 |      7 |      0.0234647   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  20 |    123 |      0.0219577   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.0145538   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0260465   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.00936893  |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.00664357  |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.0155597   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.0120311   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.0185452   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.0401187   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.0393849   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.0455094   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.0192473   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.0668443   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.104841    |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.155879    |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.12929     |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.109636    |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.101293    |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.113042    |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  35 |     42 |      0.0243813   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  35 |      7 |      0.0245884   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  35 |    123 |      0.0141788   |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  35 |     42 |      0.01061     |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  35 |      7 |      0.0104574   |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  35 |    123 |      0.0142654   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  35 |     42 |      0.028405    |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  35 |      7 |      0.0212136   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.0361155   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  35 |     42 |      0.0299469   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.0270239   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  35 |    123 |      0.0233127   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.0107826   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0146576   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.00818     |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.0128206   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0101813   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.0117386   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.0135835   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.0254556   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.0250972   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.0474993   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.010392    |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.0133291   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.105913    |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.150172    |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.0858618   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.103283    |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0931934   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.131546    |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  55 |     42 |      0.0277115   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  55 |      7 |      0.0238749   |
| GS       | swa_bestsigma_ext       | state_cond                  |   1 |  55 |    123 |      0.0156861   |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  55 |     42 |      0.00657234  |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  55 |      7 |      0.0157486   |
| GS       | swa_bestsigma_ext       | state_free                  |   1 |  55 |    123 |      0.0159708   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  55 |     42 |      0.0383812   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  55 |      7 |      0.0242053   |
| GS       | swa_bestsigma_ext       | state_cond                  |   5 |  55 |    123 |      0.0191227   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.0195561   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.0320559   |
| GS       | swa_bestsigma_ext       | state_free                  |   5 |  55 |    123 |      0.0163957   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.0143209   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0146557   |
| GS       | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.0113979   |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.00733337  |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.01082     |
| GS       | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.00652513  |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.0355254   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.0300561   |
| GS       | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.0276409   |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.00966177  |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.01015     |
| GS       | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.0301243   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.0689183   |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.15195     |
| GS       | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.0655757   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.109029    |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0833293   |
| GS       | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.0418836   |
| GS       | conditioning_regime_ext | state_cond_full             |   1 |  55 |     42 |      0.0277115   |
| GS       | conditioning_regime_ext | state_cond_full             |   1 |  55 |      7 |      0.0238749   |
| GS       | conditioning_regime_ext | state_cond_full             |   1 |  55 |    123 |      0.0156861   |
| GS       | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.0355254   |
| GS       | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.0300561   |
| GS       | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.0276409   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |     42 |      0.0440246   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |      7 |      0.0342747   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |    123 |      0.0350638   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0278536   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0475942   |
| GS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.0231138   |
| GS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.014237    |
| GS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.0193745   |
| GS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.017253    |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |     42 |      0.00317869  |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |      7 |      0.00991211  |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |    123 |      0.00768289  |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.00797663  |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0112701   |
| GS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0107499   |
| MS       | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.00332022  |
| MS       | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.00392199  |
| MS       | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.00146675  |
| MS       | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.00703752  |
| MS       | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0019393   |
| MS       | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.00167537  |
| MS       | baseline_ext            | state_cond                  |   5 |  20 |    123 |      0.00555137  |
| MS       | baseline_ext            | state_cond                  |  10 |  20 |     42 |      0.00960306  |
| MS       | baseline_ext            | state_free                  |  21 |  20 |      7 |      0.000983749 |
| MS       | baseline_ext            | state_cond                  |  10 |  35 |     42 |      0.00129924  |
| MS       | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.00335059  |
| MS       | baseline_ext            | state_free                  |  10 |  35 |    123 |      0.0021883   |
| MS       | baseline_ext            | state_cond                  |  21 |  35 |    123 |      0.000915147 |
| MS       | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.00519571  |
| MS       | baseline_ext            | state_cond                  |  10 |  55 |      7 |      0.000627611 |
| MS       | baseline_ext            | state_cond                  |  21 |  55 |     42 |      0.00404844  |
| MS       | baseline_ext            | state_free                  |  21 |  55 |     42 |      0.00401602  |
| MS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.00593448  |
| MS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0110624   |
| MS       | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.00643289  |
| MS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.00328827  |
| MS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.00883603  |
| MS       | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.00863445  |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.00620461  |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0223156   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0113592   |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.00322187  |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.00825525  |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.00404168  |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.00179911  |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.00170279  |
| MS       | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0102474   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      1.81518e-05 |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0121029   |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.0532458   |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.0816281   |
| MS       | swa_bestsigma_ext       | state_cond                  |   5 |  20 |    123 |      0.0176812   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.000654493 |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.010408    |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.00540784  |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.048831    |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.0525761   |
| MS       | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.0157516   |
| MS       | swa_bestsigma_ext       | state_cond                  |   5 |  35 |     42 |      0.00129709  |
| MS       | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.0058578   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.00947771  |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.0101663   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.00206456  |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.00179181  |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.0042275   |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.0180624   |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.0626968   |
| MS       | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0658892   |
| MS       | swa_bestsigma_ext       | state_cond                  |   5 |  55 |     42 |      0.0219031   |
| MS       | swa_bestsigma_ext       | state_cond                  |   5 |  55 |    123 |      0.00574693  |
| MS       | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.0151373   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.0067235   |
| MS       | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.00492439  |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0164791   |
| MS       | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.0160781   |
| MS       | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.0546661   |
| MS       | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0315233   |
| MS       | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.0067235   |
| MS       | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.00492439  |
| MS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0323301   |
| MS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0322815   |
| MS       | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.0416103   |
| MS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.0113045   |
| MS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.0207501   |
| MS       | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.0158616   |
| MS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0317784   |
| MS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0304824   |
| MS       | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0217992   |
| PNC      | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0184487   |
| PNC      | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.0252323   |
| PNC      | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.0192566   |
| PNC      | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0186456   |
| PNC      | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.0234902   |
| PNC      | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0196673   |
| PNC      | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0224289   |
| PNC      | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0208342   |
| PNC      | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0196697   |
| PNC      | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0258071   |
| PNC      | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0247904   |
| PNC      | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0216924   |
| PNC      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.035979    |
| PNC      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.0292132   |
| PNC      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0361437   |
| PNC      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0453982   |
| PNC      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0339364   |
| PNC      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0415361   |
| PNC      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0303751   |
| PNC      | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0268787   |
| PNC      | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.024824    |
| PNC      | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0354188   |
| PNC      | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0235933   |
| PNC      | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0359881   |
| PNC      | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.0240883   |
| PNC      | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0281727   |
| PNC      | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.0224377   |
| PNC      | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.0247      |
| PNC      | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.02509     |
| PNC      | baseline_ext            | state_free                  |  63 |   4 |    123 |      0.0277009   |
| PNC      | baseline_ext            | state_cond                  |   5 |  10 |    123 |      0.00519756  |
| PNC      | baseline_ext            | state_free                  |   5 |  10 |     42 |      0.0388613   |
| PNC      | baseline_ext            | state_cond                  |  10 |  10 |     42 |      0.0524128   |
| PNC      | baseline_ext            | state_cond                  |  10 |  10 |      7 |      0.0565398   |
| PNC      | baseline_ext            | state_cond                  |  10 |  10 |    123 |      0.049933    |
| PNC      | baseline_ext            | state_free                  |  10 |  10 |     42 |      0.0807786   |
| PNC      | baseline_ext            | state_free                  |  10 |  10 |      7 |      0.0300572   |
| PNC      | baseline_ext            | state_free                  |  10 |  10 |    123 |      0.0361195   |
| PNC      | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.00784305  |
| PNC      | baseline_ext            | state_free                  |  21 |  10 |     42 |      0.00327638  |
| PNC      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.0113776   |
| PNC      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.0125478   |
| PNC      | baseline_ext            | state_cond                  |  63 |  10 |    123 |      0.0114422   |
| PNC      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.0153561   |
| PNC      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.00887206  |
| PNC      | baseline_ext            | state_cond                  |   5 |  20 |     42 |      0.00449375  |
| PNC      | baseline_ext            | state_cond                  |   5 |  20 |      7 |      0.00632862  |
| PNC      | baseline_ext            | state_free                  |   5 |  20 |     42 |      0.00212983  |
| PNC      | baseline_ext            | state_free                  |   5 |  20 |      7 |      0.0194748   |
| PNC      | baseline_ext            | state_cond                  |  10 |  20 |    123 |      0.00857667  |
| PNC      | baseline_ext            | state_free                  |  10 |  20 |     42 |      0.00632766  |
| PNC      | baseline_ext            | state_free                  |  10 |  20 |      7 |      0.0308631   |
| PNC      | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.0183339   |
| PNC      | baseline_ext            | state_cond                  |  21 |  20 |     42 |      0.00996927  |
| PNC      | baseline_ext            | state_cond                  |  21 |  20 |      7 |      0.0119007   |
| PNC      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.0063465   |
| PNC      | baseline_ext            | state_free                  |  21 |  20 |     42 |      0.00914911  |
| PNC      | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.0114067   |
| PNC      | baseline_ext            | state_cond                  |  63 |  20 |      7 |      0.00289753  |
| PNC      | baseline_ext            | state_cond                  |  63 |  20 |    123 |      0.00238851  |
| PNC      | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.00843147  |
| PNC      | baseline_ext            | state_free                  |   1 |  35 |    123 |      0.00496564  |
| PNC      | baseline_ext            | state_cond                  |   5 |  35 |     42 |      0.00234375  |
| PNC      | baseline_ext            | state_cond                  |   5 |  35 |      7 |      0.0047842   |
| PNC      | baseline_ext            | state_cond                  |   5 |  35 |    123 |      0.0101079   |
| PNC      | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.00329146  |
| PNC      | baseline_ext            | state_free                  |   5 |  35 |    123 |      0.00659499  |
| PNC      | baseline_ext            | state_cond                  |  10 |  35 |     42 |      0.00759397  |
| PNC      | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.0112084   |
| PNC      | baseline_ext            | state_cond                  |  10 |  35 |    123 |      0.0125652   |
| PNC      | baseline_ext            | state_free                  |  10 |  35 |     42 |      0.0186211   |
| PNC      | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.0119782   |
| PNC      | baseline_ext            | state_cond                  |  21 |  35 |      7 |      0.00908599  |
| PNC      | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.00517569  |
| PNC      | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.00787125  |
| PNC      | baseline_ext            | state_free                  |  63 |  35 |     42 |      0.00450049  |
| PNC      | baseline_ext            | state_free                  |  63 |  35 |    123 |      0.00470004  |
| PNC      | baseline_ext            | state_cond                  |   5 |  55 |     42 |      0.0120827   |
| PNC      | baseline_ext            | state_cond                  |   5 |  55 |      7 |      0.0122171   |
| PNC      | baseline_ext            | state_cond                  |   5 |  55 |    123 |      0.00734196  |
| PNC      | baseline_ext            | state_free                  |   5 |  55 |      7 |      0.00601396  |
| PNC      | baseline_ext            | state_free                  |   5 |  55 |    123 |      0.00117502  |
| PNC      | baseline_ext            | state_cond                  |  10 |  55 |    123 |      0.0539933   |
| PNC      | baseline_ext            | state_free                  |  10 |  55 |      7 |      0.0444075   |
| PNC      | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.0307332   |
| PNC      | baseline_ext            | state_free                  |  21 |  55 |    123 |      0.0224606   |
| PNC      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.00193081  |
| PNC      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.00482282  |
| PNC      | baseline_ext            | state_free                  |  63 |  55 |     42 |      0.00301895  |
| PNC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.0124955   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0152283   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.0172818   |
| PNC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.0159656   |
| PNC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.0176629   |
| PNC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.021095    |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.0261046   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.0205357   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0240006   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0208516   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.0205172   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0287414   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0336474   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0355239   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0348698   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0337107   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.034762    |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.034597    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0443034   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0400808   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0400457   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.041734    |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0500629   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0394509   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0514154   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0411034   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0469599   |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0530573   |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0535907   |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0408124   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  10 |    123 |      0.00208405  |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.0321558   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.0358055   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.0369308   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.0299378   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.0382445   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.0382674   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |     42 |      0.0207012   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.0241595   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.0259807   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.029167    |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.0326867   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.0256882   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0987139   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.121981    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.110158    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.116781    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.0989731   |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.0882397   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |     42 |      0.00655225  |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |      7 |      0.00328258  |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |    123 |      0.00490407  |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  20 |      7 |      0.00411228  |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.0446186   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0472586   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.039232    |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.031014    |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.0387864   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.0445063   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.0473688   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.0583585   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.0541029   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.0459385   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.0418623   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.0514601   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.151071    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.1725      |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.146014    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.124364    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.121427    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.12187     |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |     42 |      0.000882722 |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |      7 |      0.0100323   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.0131236   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.00408492  |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  35 |    123 |      0.0057498   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.0385495   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0357865   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.0473171   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.0321766   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0421558   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.0464939   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.0459679   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.0444046   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.0596338   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.0445274   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.0655769   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.0541042   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.138732    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.156334    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.138746    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.119353    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.131776    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.125141    |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |     42 |      0.0136915   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |      7 |      0.0176464   |
| PNC      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |    123 |      0.0156299   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.0010148   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.0114418   |
| PNC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |    123 |      0.0128528   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.0310451   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0511671   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.046887    |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.0349713   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.0417405   |
| PNC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.0379125   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.0487324   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.0630237   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.0638462   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0499197   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.0529424   |
| PNC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.0594016   |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.105204    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.122425    |
| PNC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.117902    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.127279    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.108813    |
| PNC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.10105     |
| PNC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.0487324   |
| PNC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.0630237   |
| PNC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.0638462   |
| PNC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0263434   |
| PNC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0370947   |
| PNC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.0227376   |
| PNC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.0244432   |
| PNC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.025231    |
| PNC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.0247494   |
| PNC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0179125   |
| PNC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.011387    |
| PNC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0116673   |
| USB      | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0273      |
| USB      | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.0322731   |
| USB      | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.0301216   |
| USB      | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0379969   |
| USB      | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.0318633   |
| USB      | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0315065   |
| USB      | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.033459    |
| USB      | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0355691   |
| USB      | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0305016   |
| USB      | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0392315   |
| USB      | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0317602   |
| USB      | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0304976   |
| USB      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.038787    |
| USB      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.0345867   |
| USB      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0336632   |
| USB      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0403125   |
| USB      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0434173   |
| USB      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0340425   |
| USB      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0806236   |
| USB      | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0551401   |
| USB      | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.0569321   |
| USB      | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0505668   |
| USB      | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0534278   |
| USB      | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0490989   |
| USB      | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.0715187   |
| USB      | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0366646   |
| USB      | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.0543264   |
| USB      | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.0443419   |
| USB      | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.104223    |
| USB      | baseline_ext            | state_free                  |  63 |   4 |    123 |      0.064142    |
| USB      | baseline_ext            | state_cond                  |   1 |  10 |     42 |      0.0410254   |
| USB      | baseline_ext            | state_cond                  |   1 |  10 |      7 |      0.0329807   |
| USB      | baseline_ext            | state_cond                  |   1 |  10 |    123 |      0.0310867   |
| USB      | baseline_ext            | state_free                  |   1 |  10 |     42 |      0.0359669   |
| USB      | baseline_ext            | state_free                  |   1 |  10 |      7 |      0.0417462   |
| USB      | baseline_ext            | state_free                  |   1 |  10 |    123 |      0.0480867   |
| USB      | baseline_ext            | state_cond                  |   5 |  10 |     42 |      0.0504449   |
| USB      | baseline_ext            | state_cond                  |   5 |  10 |      7 |      0.0445476   |
| USB      | baseline_ext            | state_cond                  |   5 |  10 |    123 |      0.0363829   |
| USB      | baseline_ext            | state_free                  |   5 |  10 |     42 |      0.0369356   |
| USB      | baseline_ext            | state_free                  |   5 |  10 |      7 |      0.0232008   |
| USB      | baseline_ext            | state_free                  |   5 |  10 |    123 |      0.0321091   |
| USB      | baseline_ext            | state_cond                  |  10 |  10 |     42 |      0.0761717   |
| USB      | baseline_ext            | state_cond                  |  10 |  10 |      7 |      0.069381    |
| USB      | baseline_ext            | state_cond                  |  10 |  10 |    123 |      0.0488429   |
| USB      | baseline_ext            | state_free                  |  10 |  10 |     42 |      0.0796249   |
| USB      | baseline_ext            | state_free                  |  10 |  10 |      7 |      0.0953412   |
| USB      | baseline_ext            | state_free                  |  10 |  10 |    123 |      0.0432849   |
| USB      | baseline_ext            | state_cond                  |  21 |  10 |     42 |      0.0787893   |
| USB      | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.108305    |
| USB      | baseline_ext            | state_cond                  |  21 |  10 |    123 |      0.0923159   |
| USB      | baseline_ext            | state_free                  |  21 |  10 |     42 |      0.0909448   |
| USB      | baseline_ext            | state_free                  |  21 |  10 |      7 |      0.110489    |
| USB      | baseline_ext            | state_free                  |  21 |  10 |    123 |      0.0822575   |
| USB      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.0131882   |
| USB      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.0121808   |
| USB      | baseline_ext            | state_cond                  |  63 |  10 |    123 |      0.0087817   |
| USB      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.00180248  |
| USB      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.0893462   |
| USB      | baseline_ext            | state_cond                  |   1 |  20 |     42 |      0.0525287   |
| USB      | baseline_ext            | state_cond                  |   1 |  20 |      7 |      0.0576246   |
| USB      | baseline_ext            | state_cond                  |   1 |  20 |    123 |      0.0532887   |
| USB      | baseline_ext            | state_free                  |   1 |  20 |     42 |      0.0433636   |
| USB      | baseline_ext            | state_free                  |   1 |  20 |      7 |      0.0496827   |
| USB      | baseline_ext            | state_free                  |   1 |  20 |    123 |      0.0459502   |
| USB      | baseline_ext            | state_cond                  |   5 |  20 |     42 |      0.0366705   |
| USB      | baseline_ext            | state_cond                  |   5 |  20 |      7 |      0.0448032   |
| USB      | baseline_ext            | state_cond                  |   5 |  20 |    123 |      0.0214658   |
| USB      | baseline_ext            | state_free                  |   5 |  20 |     42 |      0.0528562   |
| USB      | baseline_ext            | state_free                  |   5 |  20 |      7 |      0.0237012   |
| USB      | baseline_ext            | state_free                  |   5 |  20 |    123 |      0.0364611   |
| USB      | baseline_ext            | state_cond                  |  10 |  20 |     42 |      0.0900531   |
| USB      | baseline_ext            | state_cond                  |  10 |  20 |      7 |      0.0489462   |
| USB      | baseline_ext            | state_cond                  |  10 |  20 |    123 |      0.0582111   |
| USB      | baseline_ext            | state_free                  |  10 |  20 |     42 |      0.0788768   |
| USB      | baseline_ext            | state_free                  |  10 |  20 |      7 |      0.0579565   |
| USB      | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.0110536   |
| USB      | baseline_ext            | state_cond                  |  21 |  20 |     42 |      0.10553     |
| USB      | baseline_ext            | state_cond                  |  21 |  20 |      7 |      0.0955768   |
| USB      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.125103    |
| USB      | baseline_ext            | state_free                  |  21 |  20 |     42 |      0.0196073   |
| USB      | baseline_ext            | state_free                  |  21 |  20 |      7 |      0.0869592   |
| USB      | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.0273657   |
| USB      | baseline_ext            | state_cond                  |  63 |  20 |     42 |      0.10633     |
| USB      | baseline_ext            | state_cond                  |  63 |  20 |    123 |      0.104435    |
| USB      | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.0658007   |
| USB      | baseline_ext            | state_free                  |  63 |  20 |    123 |      0.0752578   |
| USB      | baseline_ext            | state_cond                  |   1 |  35 |     42 |      0.0544816   |
| USB      | baseline_ext            | state_cond                  |   1 |  35 |      7 |      0.0512236   |
| USB      | baseline_ext            | state_cond                  |   1 |  35 |    123 |      0.0694538   |
| USB      | baseline_ext            | state_free                  |   1 |  35 |     42 |      0.0711626   |
| USB      | baseline_ext            | state_free                  |   1 |  35 |      7 |      0.0847779   |
| USB      | baseline_ext            | state_free                  |   1 |  35 |    123 |      0.0707935   |
| USB      | baseline_ext            | state_cond                  |   5 |  35 |     42 |      0.0276741   |
| USB      | baseline_ext            | state_cond                  |   5 |  35 |      7 |      0.0334612   |
| USB      | baseline_ext            | state_cond                  |   5 |  35 |    123 |      0.0257858   |
| USB      | baseline_ext            | state_free                  |   5 |  35 |     42 |      0.0665208   |
| USB      | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.0604089   |
| USB      | baseline_ext            | state_free                  |   5 |  35 |    123 |      0.0798992   |
| USB      | baseline_ext            | state_cond                  |  10 |  35 |     42 |      0.0583354   |
| USB      | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.0508901   |
| USB      | baseline_ext            | state_cond                  |  10 |  35 |    123 |      0.0375863   |
| USB      | baseline_ext            | state_free                  |  10 |  35 |     42 |      0.0920313   |
| USB      | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.0663708   |
| USB      | baseline_ext            | state_free                  |  10 |  35 |    123 |      0.0890389   |
| USB      | baseline_ext            | state_cond                  |  21 |  35 |     42 |      0.0947338   |
| USB      | baseline_ext            | state_cond                  |  21 |  35 |      7 |      0.0980998   |
| USB      | baseline_ext            | state_cond                  |  21 |  35 |    123 |      0.0786944   |
| USB      | baseline_ext            | state_free                  |  21 |  35 |     42 |      0.0126434   |
| USB      | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.111038    |
| USB      | baseline_ext            | state_free                  |  21 |  35 |    123 |      0.0863896   |
| USB      | baseline_ext            | state_cond                  |  63 |  35 |     42 |      0.064823    |
| USB      | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.060315    |
| USB      | baseline_ext            | state_cond                  |  63 |  35 |    123 |      0.031742    |
| USB      | baseline_ext            | state_free                  |  63 |  35 |     42 |      0.0454432   |
| USB      | baseline_ext            | state_free                  |  63 |  35 |      7 |      0.0720626   |
| USB      | baseline_ext            | state_free                  |  63 |  35 |    123 |      0.0246907   |
| USB      | baseline_ext            | state_cond                  |   1 |  55 |     42 |      0.0493308   |
| USB      | baseline_ext            | state_cond                  |   1 |  55 |      7 |      0.0540415   |
| USB      | baseline_ext            | state_cond                  |   1 |  55 |    123 |      0.0488034   |
| USB      | baseline_ext            | state_free                  |   1 |  55 |     42 |      0.0917784   |
| USB      | baseline_ext            | state_free                  |   1 |  55 |      7 |      0.0835691   |
| USB      | baseline_ext            | state_free                  |   1 |  55 |    123 |      0.0356146   |
| USB      | baseline_ext            | state_cond                  |   5 |  55 |     42 |      0.0352221   |
| USB      | baseline_ext            | state_cond                  |   5 |  55 |      7 |      0.0700236   |
| USB      | baseline_ext            | state_cond                  |   5 |  55 |    123 |      0.0483771   |
| USB      | baseline_ext            | state_free                  |   5 |  55 |     42 |      0.0660263   |
| USB      | baseline_ext            | state_free                  |   5 |  55 |      7 |      0.0459929   |
| USB      | baseline_ext            | state_free                  |   5 |  55 |    123 |      0.0837689   |
| USB      | baseline_ext            | state_cond                  |  10 |  55 |     42 |      0.0903131   |
| USB      | baseline_ext            | state_cond                  |  10 |  55 |      7 |      0.0944296   |
| USB      | baseline_ext            | state_cond                  |  10 |  55 |    123 |      0.103726    |
| USB      | baseline_ext            | state_free                  |  10 |  55 |     42 |      0.105827    |
| USB      | baseline_ext            | state_free                  |  10 |  55 |      7 |      0.0689522   |
| USB      | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.0750366   |
| USB      | baseline_ext            | state_cond                  |  21 |  55 |     42 |      0.110531    |
| USB      | baseline_ext            | state_cond                  |  21 |  55 |      7 |      0.113589    |
| USB      | baseline_ext            | state_cond                  |  21 |  55 |    123 |      0.0658351   |
| USB      | baseline_ext            | state_free                  |  21 |  55 |     42 |      0.109184    |
| USB      | baseline_ext            | state_free                  |  21 |  55 |      7 |      0.117262    |
| USB      | baseline_ext            | state_free                  |  21 |  55 |    123 |      0.102854    |
| USB      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.012045    |
| USB      | baseline_ext            | state_cond                  |  63 |  55 |      7 |      0.0168196   |
| USB      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.0214339   |
| USB      | baseline_ext            | state_free                  |  63 |  55 |     42 |      0.0337525   |
| USB      | baseline_ext            | state_free                  |  63 |  55 |      7 |      0.0287643   |
| USB      | baseline_ext            | state_free                  |  63 |  55 |    123 |      0.0242783   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.029938    |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0310385   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.0309366   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.0304339   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.0327351   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.0331603   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.04377     |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.0392467   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0373147   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0384183   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.038196    |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0383726   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0441906   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0392189   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0401694   |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.042496    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0445063   |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0449073   |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.114891    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0589554   |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0789925   |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0687997   |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0697537   |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0812007   |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.150874    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0793645   |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0801292   |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0854352   |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.162342    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.118373    |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  10 |     42 |      0.0504647   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  10 |      7 |      0.0471433   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  10 |    123 |      0.0441358   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  10 |     42 |      0.040263    |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  10 |      7 |      0.0566118   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  10 |    123 |      0.0446148   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |     42 |      0.0446787   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |      7 |      0.0350154   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |    123 |      0.0306928   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |     42 |      0.0356551   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |      7 |      0.0460637   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |    123 |      0.0426131   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.117164    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.107697    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.112645    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.100285    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.138807    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.110434    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |     42 |      0.164551    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.164626    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.157723    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.156814    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.177955    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.168485    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.282498    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.287072    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.276071    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.2698      |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.250751    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.293011    |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  20 |     42 |      0.0929344   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  20 |      7 |      0.07919     |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  20 |    123 |      0.0754621   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  20 |     42 |      0.0830646   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  20 |      7 |      0.0963335   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  20 |    123 |      0.0681486   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |     42 |      0.0673588   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |      7 |      0.0647536   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |    123 |      0.0548687   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |     42 |      0.0637081   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |      7 |      0.066664    |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |    123 |      0.0756522   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.167959    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.167594    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.154791    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.143971    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.148454    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.142898    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.212919    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.241968    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.244767    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.216939    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.208644    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.225745    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.525387    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.492528    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.529594    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.506906    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.393501    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.513492    |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  35 |     42 |      0.0800105   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  35 |      7 |      0.0762213   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  35 |    123 |      0.0769454   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  35 |     42 |      0.0849746   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  35 |      7 |      0.089466    |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  35 |    123 |      0.0843075   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |     42 |      0.0676411   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |      7 |      0.0797106   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.0585977   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |     42 |      0.0763782   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.0770937   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |    123 |      0.0994814   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.176233    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.173489    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.151656    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.160691    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.157584    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.173639    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.255171    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.258584    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.275831    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.223288    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.306505    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.259254    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.579471    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.626677    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.560356    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.55664     |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.52758     |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.595972    |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  55 |     42 |      0.0927115   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  55 |      7 |      0.0931111   |
| USB      | swa_bestsigma_ext       | state_cond                  |   1 |  55 |    123 |      0.0810615   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  55 |     42 |      0.106436    |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  55 |      7 |      0.0968424   |
| USB      | swa_bestsigma_ext       | state_free                  |   1 |  55 |    123 |      0.0936066   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |     42 |      0.076903    |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |      7 |      0.0948902   |
| USB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |    123 |      0.0768801   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.0816136   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.0993496   |
| USB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |    123 |      0.0954891   |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.176803    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.206869    |
| USB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.196953    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.149772    |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.1624      |
| USB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.179914    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.249907    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.283019    |
| USB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.248039    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.275098    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.295851    |
| USB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.260702    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.568842    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.675081    |
| USB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.555229    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.609463    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.506765    |
| USB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.575338    |
| USB      | conditioning_regime_ext | state_cond_full             |   1 |  55 |     42 |      0.0927115   |
| USB      | conditioning_regime_ext | state_cond_full             |   1 |  55 |      7 |      0.0931111   |
| USB      | conditioning_regime_ext | state_cond_full             |   1 |  55 |    123 |      0.0810615   |
| USB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.249907    |
| USB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.283019    |
| USB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.248039    |
| USB      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |     42 |      0.0734158   |
| USB      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |      7 |      0.0623933   |
| USB      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |    123 |      0.0665823   |
| USB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.152667    |
| USB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.179525    |
| USB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.135621    |
| USB      | conditioning_regime_ext | state_cond_own_only         |   1 |  55 |     42 |      0.00484953  |
| USB      | conditioning_regime_ext | state_cond_own_only         |   1 |  55 |      7 |      0.00123129  |
| USB      | conditioning_regime_ext | state_cond_own_only         |   1 |  55 |    123 |      0.0015274   |
| USB      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.0553008   |
| USB      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.0523478   |
| USB      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.0515762   |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |     42 |      0.0267535   |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |      7 |      0.0569044   |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |    123 |      0.040434    |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.103514    |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0991555   |
| USB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.111305    |
| FITB     | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0274073   |
| FITB     | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.0226169   |
| FITB     | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.0139446   |
| FITB     | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0120962   |
| FITB     | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.0233822   |
| FITB     | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0179868   |
| FITB     | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0202181   |
| FITB     | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0173724   |
| FITB     | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0181357   |
| FITB     | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0217544   |
| FITB     | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0225527   |
| FITB     | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0186756   |
| FITB     | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.00553787  |
| FITB     | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.0132282   |
| FITB     | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.00203085  |
| FITB     | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0130409   |
| FITB     | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0188292   |
| FITB     | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0192542   |
| FITB     | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0181088   |
| FITB     | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0114903   |
| FITB     | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.00222195  |
| FITB     | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0185713   |
| FITB     | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.00671125  |
| FITB     | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0137081   |
| FITB     | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.00549388  |
| FITB     | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0127753   |
| FITB     | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.00589896  |
| FITB     | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.00843454  |
| FITB     | baseline_ext            | state_cond                  |  10 |  10 |     42 |      0.0015662   |
| FITB     | baseline_ext            | state_cond                  |  10 |  10 |    123 |      0.00905851  |
| FITB     | baseline_ext            | state_free                  |  10 |  10 |      7 |      0.0107425   |
| FITB     | baseline_ext            | state_free                  |  10 |  20 |      7 |      0.00305561  |
| FITB     | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.013218    |
| FITB     | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.0191772   |
| FITB     | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.0152827   |
| FITB     | baseline_ext            | state_free                  |  10 |  35 |     42 |      0.00898609  |
| FITB     | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.0223552   |
| FITB     | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.00356922  |
| FITB     | baseline_ext            | state_cond                  |  10 |  55 |     42 |      0.00545606  |
| FITB     | baseline_ext            | state_cond                  |  10 |  55 |      7 |      0.00560102  |
| FITB     | baseline_ext            | state_cond                  |  10 |  55 |    123 |      0.016297    |
| FITB     | baseline_ext            | state_free                  |  10 |  55 |     42 |      0.0101014   |
| FITB     | baseline_ext            | state_free                  |  10 |  55 |      7 |      0.014063    |
| FITB     | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.000201319 |
| FITB     | baseline_ext            | state_free                  |  21 |  55 |      7 |      0.0318428   |
| FITB     | baseline_ext            | state_free                  |  21 |  55 |    123 |      0.0197717   |
| FITB     | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.000867937 |
| FITB     | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.00537262  |
| FITB     | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.0126035   |
| FITB     | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0141678   |
| FITB     | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.0113957   |
| FITB     | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.011682    |
| FITB     | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.0166571   |
| FITB     | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.0169042   |
| FITB     | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.0226281   |
| FITB     | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.022714    |
| FITB     | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0194763   |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0214427   |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.0212685   |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0273545   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0132854   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0174497   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0133029   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0156956   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0182912   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0225323   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0235889   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0189961   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0201812   |
| FITB     | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0327735   |
| FITB     | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.023541    |
| FITB     | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0262986   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0366941   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0245787   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0356336   |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0388107   |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0197487   |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0192903   |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |  10 |      7 |      0.00533489  |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.0145653   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.013171    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.0123816   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.00371507  |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.0252219   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.00729898  |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.123386    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.133067    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.124816    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.129252    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.116976    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.108268    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.0135858   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0204921   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.0202296   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.0182064   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.0297337   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.0200725   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.186258    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.210691    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.193774    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.17297     |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.160751    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.181352    |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.00789723  |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.0120266   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0224961   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.0261012   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.0217587   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0320638   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.0251521   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.00188217  |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.152865    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.181318    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.182911    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.143245    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.169855    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.152794    |
| FITB     | swa_bestsigma_ext       | state_free                  |   5 |  55 |    123 |      0.000981901 |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.025168    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0367824   |
| FITB     | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.0466815   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.0373059   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.0382615   |
| FITB     | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.028009    |
| FITB     | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0073534   |
| FITB     | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.00252733  |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.113952    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.151179    |
| FITB     | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.145834    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.142833    |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.11823     |
| FITB     | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.116826    |
| FITB     | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.0156294   |
| FITB     | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.00302229  |
| FITB     | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.00572452  |
| FITB     | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.00806818  |
| MTB      | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0323476   |
| MTB      | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.0448512   |
| MTB      | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.0367819   |
| MTB      | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.0403177   |
| MTB      | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.0453234   |
| MTB      | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0296931   |
| MTB      | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0369825   |
| MTB      | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0397923   |
| MTB      | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0364871   |
| MTB      | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0382205   |
| MTB      | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0411711   |
| MTB      | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0399858   |
| MTB      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.0563946   |
| MTB      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.062132    |
| MTB      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0466014   |
| MTB      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0550786   |
| MTB      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0444435   |
| MTB      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0609876   |
| MTB      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0404582   |
| MTB      | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0410614   |
| MTB      | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.0447897   |
| MTB      | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0372549   |
| MTB      | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0358412   |
| MTB      | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0525857   |
| MTB      | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.050169    |
| MTB      | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.0544558   |
| MTB      | baseline_ext            | state_cond                  |  63 |   4 |    123 |      0.0507171   |
| MTB      | baseline_ext            | state_free                  |  63 |   4 |     42 |      0.0663221   |
| MTB      | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.0796182   |
| MTB      | baseline_ext            | state_free                  |  63 |   4 |    123 |      0.0625292   |
| MTB      | baseline_ext            | state_cond                  |   5 |  10 |      7 |      0.0113468   |
| MTB      | baseline_ext            | state_cond                  |   5 |  10 |    123 |      0.00701454  |
| MTB      | baseline_ext            | state_free                  |   5 |  10 |     42 |      0.0455957   |
| MTB      | baseline_ext            | state_free                  |   5 |  10 |    123 |      0.0350092   |
| MTB      | baseline_ext            | state_cond                  |  10 |  10 |     42 |      0.00552872  |
| MTB      | baseline_ext            | state_cond                  |  10 |  10 |      7 |      0.0122197   |
| MTB      | baseline_ext            | state_cond                  |  10 |  10 |    123 |      0.00559119  |
| MTB      | baseline_ext            | state_free                  |  10 |  10 |     42 |      0.00364855  |
| MTB      | baseline_ext            | state_cond                  |  21 |  10 |     42 |      0.0021179   |
| MTB      | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.0112193   |
| MTB      | baseline_ext            | state_cond                  |  21 |  10 |    123 |      0.0101295   |
| MTB      | baseline_ext            | state_free                  |  21 |  10 |     42 |      0.002682    |
| MTB      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.036557    |
| MTB      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.0195961   |
| MTB      | baseline_ext            | state_cond                  |  63 |  10 |    123 |      0.0105086   |
| MTB      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.0255039   |
| MTB      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.0184453   |
| MTB      | baseline_ext            | state_cond                  |   5 |  20 |     42 |      0.00961784  |
| MTB      | baseline_ext            | state_cond                  |   5 |  20 |      7 |      0.0103183   |
| MTB      | baseline_ext            | state_free                  |   5 |  20 |      7 |      0.0186847   |
| MTB      | baseline_ext            | state_free                  |   5 |  20 |    123 |      0.0115879   |
| MTB      | baseline_ext            | state_cond                  |  10 |  20 |     42 |      0.00687579  |
| MTB      | baseline_ext            | state_cond                  |  10 |  20 |    123 |      0.00910882  |
| MTB      | baseline_ext            | state_free                  |  10 |  20 |     42 |      0.000390325 |
| MTB      | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.00374273  |
| MTB      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.00291041  |
| MTB      | baseline_ext            | state_cond                  |  63 |  20 |      7 |      0.00873974  |
| MTB      | baseline_ext            | state_cond                  |  63 |  20 |    123 |      0.00090221  |
| MTB      | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.0100255   |
| MTB      | baseline_ext            | state_free                  |  63 |  20 |      7 |      0.00137809  |
| MTB      | baseline_ext            | state_free                  |  63 |  20 |    123 |      0.00283936  |
| MTB      | baseline_ext            | state_cond                  |   5 |  35 |    123 |      0.00692115  |
| MTB      | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.00247822  |
| MTB      | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.00802789  |
| MTB      | baseline_ext            | state_cond                  |  21 |  35 |      7 |      0.0044683   |
| MTB      | baseline_ext            | state_cond                  |  21 |  35 |    123 |      0.00124345  |
| MTB      | baseline_ext            | state_cond                  |  63 |  35 |     42 |      0.00268898  |
| MTB      | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.00720391  |
| MTB      | baseline_ext            | state_free                  |  63 |  35 |     42 |      0.00959764  |
| MTB      | baseline_ext            | state_free                  |  63 |  35 |    123 |      0.00278506  |
| MTB      | baseline_ext            | state_cond                  |   5 |  55 |    123 |      0.00160322  |
| MTB      | baseline_ext            | state_free                  |   5 |  55 |    123 |      0.00742492  |
| MTB      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.0037857   |
| MTB      | baseline_ext            | state_cond                  |  63 |  55 |      7 |      0.00230751  |
| MTB      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.0183803   |
| MTB      | baseline_ext            | state_free                  |  63 |  55 |      7 |      0.00109348  |
| MTB      | baseline_ext            | state_free                  |  63 |  55 |    123 |      0.00785742  |
| MTB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.0277619   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.0318458   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.0275455   |
| MTB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.0281646   |
| MTB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.0319641   |
| MTB      | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.0317328   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.037387    |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.0419018   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0381113   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0396736   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.0395917   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0450176   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0570207   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0567241   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0551114   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0576824   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0552132   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0567547   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0568734   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0580342   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0549266   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.048998    |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0507586   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.0570263   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0808528   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0821404   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0995051   |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0847943   |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.112198    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.101384    |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |     42 |      0.00148085  |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  10 |      7 |      0.00279787  |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |     42 |      0.0129168   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |      7 |      0.00568918  |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  10 |    123 |      0.0167604   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.0314942   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.0412169   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.0334864   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.0400691   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.0397346   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.0414267   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |     42 |      0.0549133   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.0604177   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.0397787   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |     42 |      0.0490244   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.0557785   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.0486842   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.144156    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.152424    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.136981    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.155024    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.154493    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.129281    |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |     42 |      0.00831083  |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  20 |      7 |      0.0126393   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |     42 |      0.0228467   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |      7 |      0.0183301   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  20 |    123 |      0.0150142   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.0225294   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0431095   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.0227893   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.0336435   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.0257953   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.0169154   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |     42 |      0.0542887   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |      7 |      0.0674057   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.0391844   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |     42 |      0.0455745   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.0486315   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  20 |    123 |      0.0552674   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.196187    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.250649    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.200289    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.187131    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.196562    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.186939    |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |     42 |      0.00494275  |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |      7 |      0.0121714   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.00153599  |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |     42 |      0.0148049   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.0172769   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  35 |    123 |      0.0244852   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.0240389   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0321351   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.0133167   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.0314125   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0318092   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.0353511   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |     42 |      0.0568164   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.0611228   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |    123 |      0.0508259   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.050264    |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.0651361   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.0439111   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.220823    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.253484    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.194749    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.199184    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.196594    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.183779    |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |     42 |      0.00769195  |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |      7 |      0.0243593   |
| MTB      | swa_bestsigma_ext       | state_cond                  |   5 |  55 |    123 |      0.00534162  |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.0153786   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |      7 |      0.0157319   |
| MTB      | swa_bestsigma_ext       | state_free                  |   5 |  55 |    123 |      0.0166107   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.0243464   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0404964   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.0216628   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.0413676   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.0250445   |
| MTB      | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.0208493   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.105798    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.073527    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.0786272   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0709063   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.0491587   |
| MTB      | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.0691591   |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.166649    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.221916    |
| MTB      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.177934    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.194691    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.181942    |
| MTB      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.162454    |
| MTB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.105798    |
| MTB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.073527    |
| MTB      | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.0786272   |
| MTB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0314032   |
| MTB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.072734    |
| MTB      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.019486    |
| MTB      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.00249777  |
| MTB      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.00142011  |
| MTB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0149361   |
| MTB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.00984011  |
| MTB      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0191198   |
| BAC      | baseline_ext            | state_cond                  |   1 |   4 |     42 |      0.0122795   |
| BAC      | baseline_ext            | state_cond                  |   1 |   4 |      7 |      0.00921572  |
| BAC      | baseline_ext            | state_cond                  |   1 |   4 |    123 |      0.00667024  |
| BAC      | baseline_ext            | state_free                  |   1 |   4 |     42 |      0.00665045  |
| BAC      | baseline_ext            | state_free                  |   1 |   4 |      7 |      0.012007    |
| BAC      | baseline_ext            | state_free                  |   1 |   4 |    123 |      0.0127833   |
| BAC      | baseline_ext            | state_cond                  |   5 |   4 |     42 |      0.0287874   |
| BAC      | baseline_ext            | state_cond                  |   5 |   4 |      7 |      0.0244039   |
| BAC      | baseline_ext            | state_cond                  |   5 |   4 |    123 |      0.0336074   |
| BAC      | baseline_ext            | state_free                  |   5 |   4 |     42 |      0.0342767   |
| BAC      | baseline_ext            | state_free                  |   5 |   4 |      7 |      0.0230085   |
| BAC      | baseline_ext            | state_free                  |   5 |   4 |    123 |      0.0316817   |
| BAC      | baseline_ext            | state_cond                  |  10 |   4 |     42 |      0.0175532   |
| BAC      | baseline_ext            | state_cond                  |  10 |   4 |      7 |      0.015658    |
| BAC      | baseline_ext            | state_cond                  |  10 |   4 |    123 |      0.0128448   |
| BAC      | baseline_ext            | state_free                  |  10 |   4 |     42 |      0.0149454   |
| BAC      | baseline_ext            | state_free                  |  10 |   4 |      7 |      0.0195671   |
| BAC      | baseline_ext            | state_free                  |  10 |   4 |    123 |      0.0183499   |
| BAC      | baseline_ext            | state_cond                  |  21 |   4 |     42 |      0.0349329   |
| BAC      | baseline_ext            | state_cond                  |  21 |   4 |      7 |      0.0304771   |
| BAC      | baseline_ext            | state_cond                  |  21 |   4 |    123 |      0.0259596   |
| BAC      | baseline_ext            | state_free                  |  21 |   4 |     42 |      0.0225735   |
| BAC      | baseline_ext            | state_free                  |  21 |   4 |      7 |      0.0394353   |
| BAC      | baseline_ext            | state_free                  |  21 |   4 |    123 |      0.0288675   |
| BAC      | baseline_ext            | state_cond                  |  63 |   4 |     42 |      0.00947237  |
| BAC      | baseline_ext            | state_cond                  |  63 |   4 |      7 |      0.000403408 |
| BAC      | baseline_ext            | state_free                  |  63 |   4 |      7 |      0.00421739  |
| BAC      | baseline_ext            | state_cond                  |  10 |  10 |      7 |      0.00546983  |
| BAC      | baseline_ext            | state_cond                  |  21 |  10 |      7 |      0.00392345  |
| BAC      | baseline_ext            | state_cond                  |  63 |  10 |     42 |      0.00858119  |
| BAC      | baseline_ext            | state_cond                  |  63 |  10 |      7 |      0.00334791  |
| BAC      | baseline_ext            | state_free                  |  63 |  10 |     42 |      0.0265441   |
| BAC      | baseline_ext            | state_free                  |  63 |  10 |    123 |      0.0089784   |
| BAC      | baseline_ext            | state_free                  |   5 |  20 |    123 |      0.0121122   |
| BAC      | baseline_ext            | state_cond                  |  10 |  20 |     42 |      0.0164168   |
| BAC      | baseline_ext            | state_cond                  |  10 |  20 |    123 |      0.00276021  |
| BAC      | baseline_ext            | state_free                  |  10 |  20 |      7 |      0.0233696   |
| BAC      | baseline_ext            | state_free                  |  10 |  20 |    123 |      0.00826624  |
| BAC      | baseline_ext            | state_cond                  |  21 |  20 |     42 |      0.0134299   |
| BAC      | baseline_ext            | state_cond                  |  21 |  20 |    123 |      0.0183804   |
| BAC      | baseline_ext            | state_free                  |  21 |  20 |     42 |      0.00800422  |
| BAC      | baseline_ext            | state_free                  |  21 |  20 |    123 |      0.0198467   |
| BAC      | baseline_ext            | state_cond                  |  63 |  20 |     42 |      0.000629459 |
| BAC      | baseline_ext            | state_cond                  |  63 |  20 |      7 |      0.0133756   |
| BAC      | baseline_ext            | state_free                  |  63 |  20 |     42 |      0.0427432   |
| BAC      | baseline_ext            | state_free                  |  63 |  20 |      7 |      0.00156621  |
| BAC      | baseline_ext            | state_free                  |  63 |  20 |    123 |      0.0235854   |
| BAC      | baseline_ext            | state_cond                  |   5 |  35 |      7 |      0.000164367 |
| BAC      | baseline_ext            | state_free                  |   5 |  35 |      7 |      0.00229368  |
| BAC      | baseline_ext            | state_cond                  |  10 |  35 |     42 |      0.00774417  |
| BAC      | baseline_ext            | state_cond                  |  10 |  35 |      7 |      0.00424705  |
| BAC      | baseline_ext            | state_free                  |  10 |  35 |      7 |      0.0196897   |
| BAC      | baseline_ext            | state_cond                  |  21 |  35 |     42 |      0.00109706  |
| BAC      | baseline_ext            | state_cond                  |  21 |  35 |      7 |      0.0215417   |
| BAC      | baseline_ext            | state_cond                  |  21 |  35 |    123 |      0.00357161  |
| BAC      | baseline_ext            | state_free                  |  21 |  35 |     42 |      0.00971327  |
| BAC      | baseline_ext            | state_free                  |  21 |  35 |      7 |      0.0259622   |
| BAC      | baseline_ext            | state_free                  |  21 |  35 |    123 |      0.00717721  |
| BAC      | baseline_ext            | state_cond                  |  63 |  35 |      7 |      0.0248576   |
| BAC      | baseline_ext            | state_cond                  |  63 |  35 |    123 |      0.00640545  |
| BAC      | baseline_ext            | state_free                  |  63 |  35 |     42 |      0.022924    |
| BAC      | baseline_ext            | state_free                  |  63 |  35 |      7 |      0.0158497   |
| BAC      | baseline_ext            | state_free                  |  63 |  35 |    123 |      0.0379206   |
| BAC      | baseline_ext            | state_free                  |   5 |  55 |     42 |      0.0137416   |
| BAC      | baseline_ext            | state_free                  |   5 |  55 |      7 |      0.00354967  |
| BAC      | baseline_ext            | state_cond                  |  10 |  55 |     42 |      0.00962362  |
| BAC      | baseline_ext            | state_cond                  |  10 |  55 |      7 |      0.00058708  |
| BAC      | baseline_ext            | state_cond                  |  10 |  55 |    123 |      0.00238142  |
| BAC      | baseline_ext            | state_free                  |  10 |  55 |      7 |      0.00260362  |
| BAC      | baseline_ext            | state_free                  |  10 |  55 |    123 |      0.00113783  |
| BAC      | baseline_ext            | state_cond                  |  21 |  55 |      7 |      0.0108233   |
| BAC      | baseline_ext            | state_cond                  |  21 |  55 |    123 |      0.01189     |
| BAC      | baseline_ext            | state_free                  |  21 |  55 |      7 |      0.014855    |
| BAC      | baseline_ext            | state_free                  |  21 |  55 |    123 |      0.00737104  |
| BAC      | baseline_ext            | state_cond                  |  63 |  55 |     42 |      0.0423213   |
| BAC      | baseline_ext            | state_cond                  |  63 |  55 |      7 |      0.0116745   |
| BAC      | baseline_ext            | state_cond                  |  63 |  55 |    123 |      0.0356894   |
| BAC      | baseline_ext            | state_free                  |  63 |  55 |     42 |      0.0186521   |
| BAC      | baseline_ext            | state_free                  |  63 |  55 |      7 |      0.0520035   |
| BAC      | baseline_ext            | state_free                  |  63 |  55 |    123 |      0.00524578  |
| BAC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |     42 |      0.00736642  |
| BAC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |      7 |      0.00511659  |
| BAC      | swa_bestsigma_ext       | state_cond                  |   1 |   4 |    123 |      0.00591827  |
| BAC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |     42 |      0.00694991  |
| BAC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |      7 |      0.00883437  |
| BAC      | swa_bestsigma_ext       | state_free                  |   1 |   4 |    123 |      0.0097332   |
| BAC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |     42 |      0.0174134   |
| BAC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |      7 |      0.0179678   |
| BAC      | swa_bestsigma_ext       | state_cond                  |   5 |   4 |    123 |      0.0177181   |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |     42 |      0.0184685   |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |      7 |      0.0185906   |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |   4 |    123 |      0.0203445   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |     42 |      0.0212641   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |      7 |      0.0254247   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |   4 |    123 |      0.0188085   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |     42 |      0.0211569   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |      7 |      0.0261397   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |   4 |    123 |      0.0221211   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |     42 |      0.0581168   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |      7 |      0.0495827   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |   4 |    123 |      0.0480683   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |     42 |      0.0408336   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |      7 |      0.0485389   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |   4 |    123 |      0.044672    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |     42 |      0.0163255   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |      7 |      0.0247717   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |   4 |    123 |      0.0371574   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |     42 |      0.0301436   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |      7 |      0.0143521   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |   4 |    123 |      0.0377399   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |     42 |      0.00919798  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |      7 |      0.0266533   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  10 |    123 |      0.0125628   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |     42 |      0.0123396   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |      7 |      0.0112672   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  10 |    123 |      0.0119236   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |      7 |      0.000432046 |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  10 |    123 |      0.000561746 |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |      7 |      0.000565561 |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  10 |    123 |      0.000285896 |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |     42 |      0.0739129   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |      7 |      0.080427    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  10 |    123 |      0.0723696   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |     42 |      0.0736692   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |      7 |      0.0732992   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  10 |    123 |      0.0668042   |
| BAC      | swa_bestsigma_ext       | state_cond                  |   1 |  20 |    123 |      0.000844512 |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |  20 |     42 |      0.00194124  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |     42 |      0.00704626  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |      7 |      0.0280333   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  20 |    123 |      0.0176163   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |     42 |      0.0141082   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |      7 |      0.0189875   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  20 |    123 |      0.0139292   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  20 |    123 |      0.00105599  |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  20 |      7 |      0.00189832  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |     42 |      0.0878144   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |      7 |      0.111119    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  20 |    123 |      0.0973246   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |     42 |      0.0926249   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |      7 |      0.0968297   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  20 |    123 |      0.0770522   |
| BAC      | swa_bestsigma_ext       | state_cond                  |   5 |  35 |    123 |      0.00115857  |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |  35 |      7 |      0.000162698 |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |  35 |    123 |      0.00523982  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |     42 |      0.0273419   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |      7 |      0.0227991   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  35 |    123 |      0.0230306   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |     42 |      0.0202342   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |      7 |      0.0277945   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  35 |    123 |      0.0297545   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  35 |      7 |      0.00688467  |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |     42 |      0.00688777  |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |      7 |      0.0114623   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  35 |    123 |      0.00668368  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |     42 |      0.0908683   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |      7 |      0.104165    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  35 |    123 |      0.0965699   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |     42 |      0.0801481   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |      7 |      0.0915299   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  35 |    123 |      0.0769831   |
| BAC      | swa_bestsigma_ext       | state_free                  |   1 |  55 |     42 |      0.00234899  |
| BAC      | swa_bestsigma_ext       | state_free                  |   5 |  55 |     42 |      0.00770864  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |     42 |      0.0397006   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |      7 |      0.0567132   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  10 |  55 |    123 |      0.0466615   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |     42 |      0.0285054   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |      7 |      0.0233775   |
| BAC      | swa_bestsigma_ext       | state_free                  |  10 |  55 |    123 |      0.0118509   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |     42 |      0.00323782  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |      7 |      0.00706062  |
| BAC      | swa_bestsigma_ext       | state_cond                  |  21 |  55 |    123 |      0.00291738  |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |     42 |      0.0264526   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |      7 |      0.0199095   |
| BAC      | swa_bestsigma_ext       | state_free                  |  21 |  55 |    123 |      0.0139438   |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |     42 |      0.084127    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |      7 |      0.101999    |
| BAC      | swa_bestsigma_ext       | state_cond                  |  63 |  55 |    123 |      0.106961    |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |     42 |      0.0913979   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |      7 |      0.0881778   |
| BAC      | swa_bestsigma_ext       | state_free                  |  63 |  55 |    123 |      0.0704699   |
| BAC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |     42 |      0.00323782  |
| BAC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |      7 |      0.00706062  |
| BAC      | conditioning_regime_ext | state_cond_full             |  21 |  55 |    123 |      0.00291738  |
| BAC      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |      7 |      0.00677833  |
| BAC      | conditioning_regime_ext | state_cond_macro_only       |   1 |  55 |    123 |      0.0124656   |
| BAC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |     42 |      0.0241662   |
| BAC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |      7 |      0.0134192   |
| BAC      | conditioning_regime_ext | state_cond_macro_only       |  21 |  55 |    123 |      0.00110445  |
| BAC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |     42 |      0.0480505   |
| BAC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |      7 |      0.0218073   |
| BAC      | conditioning_regime_ext | state_cond_own_only         |  21 |  55 |    123 |      0.0259081   |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |     42 |      0.0169154   |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |      7 |      0.00894604  |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |   1 |  55 |    123 |      0.0126091   |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |     42 |      0.0197278   |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |      7 |      0.0241738   |
| BAC      | conditioning_regime_ext | state_cond_other_banks_only |  21 |  55 |    123 |      0.0240727   |

