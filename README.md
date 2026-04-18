# Deep Markov Research

## Running the Multi-Asset Experiments (for Jesse)

### Setup

```bash
git pull

# Create a Python 3.10+ environment
conda create -n stat453 python=3.10
conda activate stat453

# Install dependencies
pip install -r requirements.txt

# Register the kernel so Jupyter can find it
python -m ipykernel install --user --name stat453 --display-name "stat453"
```

### Run

```bash
jupyter notebook "Extended Dataset Training.ipynb"
```

Open the notebook in your browser and run all cells (Cell → Run All).

The notebook runs Experiments A through E across all 10 bank stocks. Training
weights are cached — if you need to re-run after an interruption, simply run
all cells again and already-completed experiments will load from cache
automatically.

### After completion

```bash
# Stage results (CSVs, JSONs, summary.md, run.log — weights are gitignored)
git add results_multiasset/

git commit -m "Add multi-asset experiment results"
git push
```

If any experiment printed a FAILED marker or the run.log contains errors,
send the `results_multiasset/run.log` file via direct message.
