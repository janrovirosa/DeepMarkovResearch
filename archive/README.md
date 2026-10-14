# Archive

Early exploration (November 2025 to February 2026), kept for the record. Nothing here is used by the current code
and the notebooks won't run as they are: they point to old paths and to a `scripts/` package that was replaced by `markov/`.

- `initial_count_matrix/`: the first count-based transition matrix and heatmap for JPM (STAT 453 course project).
- `notebooks/`: transition-matrix notebooks, the midway report MLP, diagnostic experiments (baselines, MI
  permutation tests, multi-step evaluation, hard vs soft labels, validation-tuned baselines) and the comparison notebooks.
- `results/`: the small tables and metrics those notebooks wrote. Model weights were dropped and are in the git history.
- `results/master_grid_results_leaky_label.csv`: the first horizon x bins grid. Its label was the return from t to t+h,
  which includes the same day as the state X_t, so at h=2 the count baselines gain up to 0.3 nats by reading part of
  the label from the state. `notebooks/single_asset/1_grid.ipynb` reruns the grid with the label starting at t+1.
- `figures/`: plots from the comprehensive-analysis notebook.
