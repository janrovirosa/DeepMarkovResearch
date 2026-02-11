# Instructions for Running Comprehensive Analysis

## Step 1: Modify Your Training Notebook

Before running `ComprehensiveAnalysis.ipynb`, you need to save your trained model weights.

### Add these lines to `TransitionProbMatrix.ipynb` (or `TransitionProbMatrix_NEWDATA.ipynb`):

**Find the cell where you save `A_all`** (should be around the cell that creates the `transition_outputs` directory).

Add this code right after:

```python
# Save model weights for later analysis
torch.save(model.state_dict(), "transition_outputs/model_weights.pt")
print("Saved model weights to transition_outputs/model_weights.pt")
```

The complete cell should look like:

```python
import os

os.makedirs("transition_outputs", exist_ok=True)

# Save transition matrices
torch.save(A_all, "transition_outputs/A_all_model.pt")
np.save("transition_outputs/A_all_model.npy", A_all.numpy())

# Save model weights (ADD THIS)
torch.save(model.state_dict(), "transition_outputs/model_weights.pt")

print("Saved A_all and model weights to transition_outputs/")
```

## Step 2: Run Your Training Notebook

1. Open `TransitionProbMatrix.ipynb` (or `TransitionProbMatrix_NEWDATA.ipynb`)
2. Run all cells to train the model
3. This will create the `transition_outputs/` directory with:
   - `A_all_model.pt` - All time-varying transition matrices
   - `A_all_model.npy` - Same as above in numpy format
   - `model_weights.pt` - Trained model parameters

**Note**: Training will take longer on CPU but should still complete (maybe 20-40 minutes instead of 5-10 on GPU).

## Step 3: Run the Comprehensive Analysis

1. Open `ComprehensiveAnalysis.ipynb`
2. Set the configuration:
   - `USE_DIAGNOSTIC = False` if you used `dataset/train.csv`
   - `USE_DIAGNOSTIC = True` if you used `dataset/train_diagnostic.csv`
3. Run all cells

The notebook will:
- ✓ Work on CPU (just slower)
- ✓ Load your trained model
- ✓ Generate all analysis plots
- ✓ Save figures to `figures/` directory

## What You'll Get

### Analysis 1: Temporal Evolution
- `figures/temporal_matrix_distances.png` - How much matrices change over time
- `figures/temporal_specific_transitions.png` - Evolution of key transitions
- `figures/regime_comparison_volatility.png` - High vs low volatility comparison
- `figures/matrix_change_vs_features.png` - Correlation with features

### Analysis 2: Feature Attribution
- `figures/feature_importance_gradients.png` - Top driving features
- `figures/feature_groups_importance.png` - Macro vs fundamentals

### Analysis 3: State Persistence & Asymmetry
- `figures/state_persistence.png` - Diagonal analysis
- `figures/mean_reversion.png` - Extreme states return to center
- `figures/asymmetry_analysis.png` - Positive vs negative returns

## Expected Runtime

On **CPU**:
- Training notebook: 20-40 minutes
- Analysis notebook: 10-20 minutes

On **GPU** (if available):
- Training notebook: 5-10 minutes
- Analysis notebook: 2-5 minutes

## Troubleshooting

### Error: "A_all not found"
- Make sure you ran the training notebook first
- Check that `transition_outputs/A_all_model.pt` exists

### Error: "Model weights not found"
- Feature attribution analysis will be skipped
- Other analyses will still run
- Add the model saving code to your training notebook and re-run

### Error: "train2.csv not found"
- The notebooks reference `train2.csv` which doesn't exist
- Use `dataset/train.csv` instead (set `USE_DIAGNOSTIC = False`)
- Or use `dataset/train_diagnostic.csv` (set `USE_DIAGNOSTIC = True`)

### Out of memory
- Reduce `n_samples_grad` in the feature attribution cell
- Use smaller batch sizes
- Close other applications

## Next Steps After Analysis

Use the generated figures in your paper's Results section to support:

1. **Claim: Time-inhomogeneous dynamics**
   → Use temporal evolution plots

2. **Claim: Feature-conditioning matters**
   → Use feature attribution plots

3. **Claim: Economically meaningful**
   → Use persistence/asymmetry plots

All plots are publication-ready (300 DPI) and saved in `figures/`.
