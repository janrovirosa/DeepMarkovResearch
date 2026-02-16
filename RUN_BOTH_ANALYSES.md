# How to Generate Both Sets of Figures

The `ComprehensiveAnalysis.ipynb` notebook now automatically saves figures to different folders based on the dataset.

## 📊 Quick Instructions:

### **Run 1: Baseline Dataset Analysis**

1. Open `ComprehensiveAnalysis.ipynb`
2. Set configuration:
   ```python
   USE_DIAGNOSTIC = False
   ```
3. **Run All Cells**
4. ✅ Figures saved to: `figures/baseline/`

---

### **Run 2: Diagnostic Dataset Analysis**

1. In same notebook, change configuration:
   ```python
   USE_DIAGNOSTIC = True
   ```
2. **Run All Cells**
3. ✅ Figures saved to: `figures/diagnostic/`

---

## 📁 Final Structure:

```
figures/
├── baseline/          ← 4,181 samples, 113 features
│   ├── temporal_matrix_distances.png
│   ├── temporal_specific_transitions.png
│   ├── regime_comparison_volatility.png
│   ├── matrix_change_vs_features.png
│   ├── feature_importance_gradients.png
│   ├── feature_groups_importance.png
│   ├── state_persistence.png
│   ├── mean_reversion.png
│   └── asymmetry_analysis.png
│
└── diagnostic/        ← 2,368 samples, 195 features
    ├── temporal_matrix_distances.png
    ├── temporal_specific_transitions.png
    ├── regime_comparison_volatility.png
    ├── matrix_change_vs_features.png
    ├── feature_importance_gradients.png
    ├── feature_groups_importance.png
    ├── state_persistence.png
    ├── mean_reversion.png
    └── asymmetry_analysis.png
```

---

## ⏱️ Time Required:

- Baseline analysis: ~10 minutes
- Diagnostic analysis: ~10 minutes
- **Total: ~20 minutes**

---

## 💡 For Your Paper:

You can now compare:
- **Baseline**: More data (4,181 samples) but fewer features (113)
- **Diagnostic**: Less data (2,368 samples) but richer features (195 banking ratios)

This lets you show that your framework works with both dataset configurations!
