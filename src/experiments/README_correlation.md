# Case Align vs Traditional Robustness Metrics Correlation Experiment

This experiment analyzes the correlation between case align values (using the like-only variant) and traditional robustness metrics.

## Files

1. **`case_align_correlation.py`** - Main experiment script
2. **`test_correlation.py`** - Simple test script to verify setup
3. **`case_align_correlation_analysis.ipynb`** - Interactive Jupyter notebook for analysis

## Key Features

### Case Align (Like-Only Variant)
- Uses `like_only=True` in `RobustnessCBR`
- Computes `S_plus` and `R_bounded` metrics
- Focuses only on like neighbors, ignoring unlike neighbors

### Traditional Robustness Metrics
1. **Local Lipschitz Continuity** - Max ratio of explanation change to input change
2. **Local Smoothness** - Average explanation distance in epsilon-neighborhood  
3. **Local Stability** - Standard deviation of explanations in neighborhood
4. **K-Nearest Robustness** - Average explanation distance to k-NN
5. **Explanation Variance Ratio** - Local vs global explanation variance

## Quick Start

### 1. Test Setup
```bash
cd src
python experiments/test_correlation.py
```

### 2. Run Full Experiment
```bash
cd src
python experiments/case_align_correlation.py --dataset adult --n_samples 200
```

### 3. Interactive Analysis
```bash
cd src/experiments
jupyter notebook case_align_correlation_analysis.ipynb
```

## Command Line Options

```bash
python experiments/case_align_correlation.py \
  --dataset adult \
  --split test \
  --n_samples 200 \
  --epsilon 0.1 \
  --k 5 \
  --seed 42 \
  --sim_metric gower \
  --output results/correlation_results.csv
```

### Parameters
- `--dataset`: Dataset name (adult, cancer, wine, etc.)
- `--split`: Data split to use (test, train, val)
- `--n_samples`: Number of samples to evaluate
- `--epsilon`: Neighborhood radius for traditional metrics
- `--k`: Number of neighbors for case align and k-NN metrics
- `--seed`: Random seed for reproducibility
- `--sim_metric`: Similarity metric (gower, cosine, spearman)
- `--output`: Output CSV file path

## Output

The experiment produces:
1. **Detailed results CSV** - Individual sample metrics
2. **Correlation summary CSV** - Pearson and Spearman correlations
3. **Console output** - Summary statistics and key findings

### Key Metrics Analyzed
- Correlation coefficients (Pearson & Spearman)
- Significance tests (p-values)
- Distribution analysis by class
- Statistical tests between classes

## Expected Insights

### Potential Correlations
- **Negative correlation** with smoothness/stability metrics (higher case align = lower traditional robustness)
- **Positive correlation** with Lipschitz continuity in some cases
- **Class-dependent** relationships

### Interpretation
- Strong correlations suggest case align captures similar robustness concepts
- Weak correlations may indicate case align measures different aspects
- Class differences could reveal bias or differential behavior

## Troubleshooting

### Common Issues
1. **Missing explanations**: The script falls back to using input features (X) as explanations
2. **Infinite values**: Automatically handled by filtering for correlation analysis  
3. **Small neighborhoods**: Epsilon might be too small; try increasing it
4. **Memory issues**: Reduce `n_samples` for large datasets

### Dependencies
Ensure these packages are installed (see `requirements.txt`):
- numpy, pandas, scipy
- scikit-learn 
- torch (for loading .pt files)
- matplotlib, seaborn (for notebook)

## Customization

### Adding New Traditional Metrics
Modify `TraditionalRobustnessMetrics` class in `case_align_correlation.py`:

```python
def my_robustness_metric(self, X, explanations, index):
    # Your custom implementation
    return robustness_score
```

### Different Case Align Variants
Change the `RobustnessCBR` initialization:
```python
case_align = RobustnessCBR(
    like_only=False,  # Use full case align
    robust_mode="ratio",  # Or "geom"
    # ... other parameters
)
```

## Expected Runtime
- **Test**: ~10 seconds 
- **200 samples**: ~2-5 minutes
- **1000+ samples**: ~10-30 minutes (depends on dataset size)

The runtime scales with `n_samples × dataset_size` for neighborhood computations.