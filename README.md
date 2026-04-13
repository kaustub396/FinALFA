# FinALFA: Learning Optimal Sentiment-Fundamental Fusion Weights for Directional Prediction of Indian Equity Market Movements Under Strict Temporal Constraints

FinALFA is a compact research implementation for directional market prediction under a small-sample, time-ordered setting.  
The core idea is simple: instead of using a high-parameter fusion network, we learn an interpretable fusion weight (`alpha`) that balances sentiment and fundamental signals, then evaluate it with strict nested LOOCV.

## What this project contains

- `src/extract_afm_data.py`  
  Builds the AFM-ready datasets (fusion features and target labels).

- `src/add_afm_cells.py`  
  Adds AFM experiment cells to the notebook workflow.

- `notebooks/Main_Project.ipynb`  
  Main experiment notebook, including AFM analysis and visual outputs.

- `data/final_df.csv`  
  Final modeling table used in AFM experiments.

- `data/phase3_fused_signal.csv`  
  Intermediate fused sentiment-fundamental signal data.

## Method summary (AFM)

1. Build yearly sentiment and fundamental features.
2. Learn fusion signal:  
   `x_t = alpha * sentiment_t + (1 - alpha) * fundamental_t`
3. Use L2-regularized logistic regression for directional prediction.
4. Use nested LOOCV:
   - inner loop selects best `alpha`
   - outer loop evaluates generalization

This setup keeps the model small, interpretable, and less prone to overfitting in low-data regimes.

## Outputs

Typical outputs from the notebook flow:

- best alpha from landscape search
- nested LOOCV metrics (accuracy, precision, recall, F1)
- confusion matrix and model comparison plots
- full comparison table export (`full_model_comparison.csv`)

## Repro steps

1. Open `notebooks/Main_Project.ipynb`.
2. Ensure required Python packages are available:
   - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
3. Run notebook top-to-bottom.
4. Exported tables/plots are produced by notebook cells.

## Notes

- This repo is focused on the AFM stage and its data flow.
- The intent is reproducible research code, not a production trading system.
- Temporal ordering is preserved throughout evaluation to avoid leakage.
