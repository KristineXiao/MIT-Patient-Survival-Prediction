# Training Guide

## Quick Start - Train Individual Models

Instead of training all models at once, you can now train them individually:

```bash
# Train logistic regression (fastest)
python src/train_single.py --model log_reg

# Train random forest
python src/train_single.py --model random_forest

# Train gradient boosting
python src/train_single.py --model grad_boost

# Train XGBoost (requires xgboost package)
python src/train_single.py --model xgboost

# Train LightGBM (requires lightgbm package)
python src/train_single.py --model lightgbm
```

## Hyperparameter Optimization Improvements

### Training Time Reduction (with 5-fold CV):

| Model | Before | After | Reduction |
|-------|--------|-------|-----------|
| Logistic Regression | 30 fits | 30 fits | - |
| Random Forest | 360 fits | 120 fits | **67% faster** |
| Gradient Boosting | 270 fits | 40 fits | **85% faster** |
| XGBoost | 360 fits | 80 fits | **78% faster** |
| LightGBM | 540 fits | 80 fits | **85% faster** |

The optimized grids focus on the most impactful hyperparameters while removing redundant combinations.

## Original Training (All Models)

To train all models at once (will take longer):

```bash
python src/train.py
```

## Confusion Matrix Analysis

### Random Forest Results:
- **Specificity: 98.4%** - Excellent at identifying survivors (class 0)
- **Sensitivity: 27.1%** - Poor at identifying deaths (class 1)
- **Issue**: Heavily biased toward predicting survival, missing most actual deaths
- **Clinical Impact**: Not suitable if detecting deaths is critical

### Logistic Regression Results:
- **Specificity: 79.7%** - Good at identifying survivors
- **Sensitivity: 77.9%** - Much better at identifying deaths
- **Better Balance**: More clinically useful for mortality prediction
- **Trade-off**: More false alarms but catches most true deaths

### Recommendations:
1. **For mortality prediction**: Logistic regression is more appropriate due to better recall on deaths
2. **Class imbalance issue**: Consider using `class_weight='balanced'` or resampling techniques
3. **Threshold tuning**: Adjust prediction threshold to optimize sensitivity/specificity trade-off
4. **Evaluation metric**: Use ROC-AUC or PR-AUC instead of accuracy for imbalanced data

## Next Steps

1. Start with logistic regression (fastest, ~2-3 minutes)
2. Try tree-based models if you need better performance
3. Compare results using `results/metrics.csv`
4. Adjust class weights if one class is more important
