# Deep Markov Research

Neural networks used only to fill in time-varying Markov transition matrices for daily stock returns.

Counting transitions between fine return bins breaks down: with 55 bins, almost every cell of the 55 x 55 count
matrix has fewer than 5 observations. Instead, an MLP takes the current state X_t (one-hot) and a feature vector
F_t and outputs a distribution over the next bin. Evaluating it for every state with the same F_t gives a full
stochastic matrix A_t for that day, so classical tools still apply: heatmaps, row heterogeneity (does the current
state matter), row entropy, the Dobrushin coefficient, and Chapman-Kolmogorov checks that compare a direct h-step
matrix with the product of h one-step matrices.

Jan Rovirosa and Jesse Schmolze, University of Wisconsin-Madison. The project started as a STAT 453 course project.

- [Learning Time-Inhomogeneous Markov Dynamics in Financial Time Series via Neural Parameterization](https://arxiv.org/abs/2605.04690) (JPM, return state)
- [Inspectable Neural Markov Models for Non-Stationary Time Series](https://arxiv.org/abs/2605.30943) (10 US banks, volatility state)

## Layout

```
markov/                    data, models, training, metrics and plotting used everywhere
notebooks/single_asset/    paper 1, JPM, runs on a laptop CPU
  1_grid                   horizon x bins grid vs count baselines
  2_operator_experiments   sparsity, CK check, operator diagnostics, calibration, seeds and bootstrap, ablations
  3_volatility             diagnostics vs realized variance and VIX, snapshot figures
  4_signal_search          sigma, budget, feature-count and seed sweeps at h=10
notebooks/multi_asset/     paper 2 and the multi-asset experiments, plots only
  volatility               vol state vs return state
  figures                  return-state grid, sigma sweep, k-step history, feature subsets, cross-asset synchrony
experiments/               GPU jobs that write results/multi_asset and results/volatility
  prepare_banks.py         builds data/banks from data/banks_raw
  run_multiasset.py        experiments A to F
  run_vol.py               volatility-state experiments
data/                      jpm (single asset features, built by dataset.Rmd from jpm_raw), banks, banks_raw
results/                   tables, figures, per-day operator diagnostics and CK series, a few model weights
archive/                   early exploration, see archive/README.md
```

## Running

```
pip install -r requirements.txt
jupyter lab notebooks/single_asset
```

All six notebooks load saved weights or saved per-day results and only train what is missing, so together they
run in about two minutes on a CPU. Delete a `cache` folder to retrain those models from scratch.

The multi-asset experiments train about 5,500 models and need a GPU (CUDA, Apple MPS or DirectML):

```
python experiments/run_vol.py            # or: python experiments/run_vol.py JPM WFC
python experiments/run_multiasset.py     # or one step: A, B, C, D, E, F
```

Their weights are not tracked, only the one model figure 12 needs (`results/multi_asset/weights`). Everything they
produce is in `results/`, so the notebooks don't need them. Weights put in `results/volatility/_cache` or
`results/multi_asset/_cache` are loaded instead of retrained.

## Conventions

- Splits are chronological 70/15/15. Bin edges, feature scaling and baseline tuning use train (and validation) only.
- X_t is the bin of the return from day t to t+1. The h-step label is the return from t+1 to t+1+h, so it never
  shares a day with the state. The `date` column of every per-day series is the day X_t becomes known.
- Training uses a Gaussian soft label over neighbouring bins (KL loss), AdamW, early stopping on validation loss,
  and optionally an average of the last 10 epochs' weights (SWA).
- In the multi-asset results `delta_marginal` is test NLL minus log(N), the NLL of a uniform forecast over the N
  quantile bins. Negative is better.
