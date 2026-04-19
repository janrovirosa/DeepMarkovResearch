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
