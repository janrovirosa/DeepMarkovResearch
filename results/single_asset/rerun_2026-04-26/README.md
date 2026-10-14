# Rerun of the operator experiments, 2026-04-26

A full rerun of `notebooks/single_asset/2_operator_experiments` on a different machine (Windows, GPU), with the same
code, data and seeds as `paper_run`. Its weights were not kept. The models it trained are not the same as in
`paper_run` and several numbers move a lot, for example the mean Dobrushin coefficient of the 1-step state-conditioned
operator (0.012 here vs 0.029) and the h=5 Chapman-Kolmogorov KL (state-conditioned 0.021 vs state-free 0.061 here,
0.140 vs 0.022 in `paper_run`). Single-seed operator diagnostics are sensitive to the training run.
