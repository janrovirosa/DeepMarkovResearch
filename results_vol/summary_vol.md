# Volatility-State Experiment Summary
## Table 1: Mean NLL (h=1, N=55, all tickers)
| model              |   nll_test |   delta_marginal |
|:-------------------|-----------:|-----------------:|
| state_cond_macro   |     3.9938 |          -0.0135 |
| state_cond_return  |     3.9942 |          -0.0131 |
| state_cond_vol_w10 |     3.9864 |          -0.0209 |
| state_cond_vol_w21 |     3.9847 |          -0.0226 |
| state_cond_vol_w5  |     3.9955 |          -0.0119 |
| state_free         |     3.993  |          -0.0143 |

## Table 2: Mean NLL by horizon (N=55, all tickers)
| model              |      1 |      2 |      3 |      5 |
|:-------------------|-------:|-------:|-------:|-------:|
| state_cond_macro   | 3.9938 | 4.0091 | 4.0185 | 4.0126 |
| state_cond_return  | 3.9942 | 4.013  | 4.0168 | 4.0162 |
| state_cond_vol_w10 | 3.9864 | 4.0057 | 4.0133 | 4.009  |
| state_cond_vol_w21 | 3.9847 | 4.004  | 4.0089 | 4.0078 |
| state_cond_vol_w5  | 3.9955 | 4.0101 | 4.0181 | 4.0164 |
| state_free         | 3.993  | 4.011  | 4.0188 | 4.0173 |

## Table 3: Mean NLL by N bins (h=1, all tickers)
| model              |     35 |     45 |     55 |
|:-------------------|-------:|-------:|-------:|
| state_cond_macro   | 3.5394 | 3.7926 | 3.9938 |
| state_cond_return  | 3.5379 | 3.792  | 3.9942 |
| state_cond_vol_w10 | 3.5318 | 3.7843 | 3.9864 |
| state_cond_vol_w21 | 3.5285 | 3.7834 | 3.9847 |
| state_cond_vol_w5  | 3.5387 | 3.7928 | 3.9955 |
| state_free         | 3.541  | 3.7954 | 3.993  |

## Table 4: Per-ticker NLL (h=1, N=55, state_cond_return vs state_cond_vol_w21)
| ticker   |   state_cond_return |   state_cond_vol_w21 |
|:---------|--------------------:|---------------------:|
| BAC      |              4.0057 |               3.9948 |
| C        |              3.97   |               3.9659 |
| FITB     |              3.9672 |               3.9644 |
| GS       |              4.0315 |               4.0163 |
| JPM      |              3.9559 |               3.9497 |
| MS       |              3.9794 |               3.9732 |
| MTB      |              3.944  |               3.9408 |
| PNC      |              3.9905 |               3.9943 |
| USB      |              4.0996 |               4.06   |
| WFC      |              3.9983 |               3.988  |

## Table 5: FAILED cells
No FAILED markers found.

