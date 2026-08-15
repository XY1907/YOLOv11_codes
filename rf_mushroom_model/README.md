# Mushroom Irrigation RF Regressor (mist-duration model)

Part of **SpotShrooms / OSIP Mushroom**. A `RandomForestRegressor` that predicts
**how long to run the misting/irrigation system (seconds)**, given live camera
counts (from the YOLO detector) and ambient sensor readings. Ported over from
the `IIICe-SP` repo's `origin/main` branch (`scripts/mushroom_rf_pipeline.py`).

> ⚠️ **Read `mushroom_rf_evaluation_report.txt` before quoting any metric below.**
> This model trains and evaluates entirely on **synthetic** data — no real hut
> sensor/camera/actuator logs exist yet. The numbers here show the pipeline is
> statistically sound and the RF genuinely beats simpler baselines on the
> labels it was given — they are **not** evidence the labels themselves match
> real-world misting needs. That requires retraining on real hut data later
> (same pipeline, unchanged, per the project's design spec).

## Contents
- `scripts/mushroom_rf_pipeline.py` — full pipeline: synthetic data generation,
  EDA + multicollinearity check, train/test split, training (with OOB
  scoring), 5-fold cross-validation vs. baselines, split veto/duration
  evaluation, permutation importance, 8 graphs, model save, and the
  `predict()` function with veto logic. Top-of-file docstring documents which
  synthetic-label rules are literature-grounded vs. business-logic design
  choices vs. uncited/unverified.
- `mushroom_rf_model.pkl` — trained model
  (`RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
  random_state=42, oob_score=True)`).
- `mushroom_training_data.csv` — the 2000-row synthetic training set
  (`np.random.seed(42)`).
- `mushroom_rf_evaluation_report.txt` — consolidated justification document:
  full assumptions/literature block + every metric below in one place.
- `graphs/` — evaluation plots (see below).

## Inputs (features)
| Feature | Meaning |
|---|---|
| `no_mushroom_sprout_count` | count of un-sprouted spots (camera) |
| `small_medium_count` | count of small/medium mushrooms (camera) |
| `mature_count` | count of mature mushrooms (camera) |
| `total_count` | sum of the three counts |
| `mature_ratio` | `mature_count / total_count` — engineered, matches the veto threshold directly |
| `rh_pct` | relative humidity, % (sensor) |
| `temp_c` | temperature, °C (sensor) |
| `co2_ppm` | CO₂ concentration, ppm (sensor) |

## Output
`mist_duration_seconds` — seconds to run the misting system.

## Veto logic (applied around the RF prediction, in `predict()`)
1. `total_count == 0` → 0s, no mushrooms detected.
2. `mature_count / total_count >= 5%` → 0s, **alert owner, skip cycle**.
3. `rh_pct >= 92%` → 0s, humidity already saturated.
4. `co2_ppm > 2000` → 0s, ventilate first.

Otherwise, the RF model predicts the misting duration from all 8 features.

## Evaluation — the real justification (see `mushroom_rf_evaluation_report.txt` for full detail)
| Check | Result |
|---|---|
| Single 80/20 split | R²=0.972, MAE=1.14s, RMSE=1.57s |
| Out-of-bag score (bagging-internal, no held-out split needed) | R²=0.957 |
| **5-fold CV** (mean ± std) | **R²=0.960 ± 0.020**, MAE=1.18s ± 0.09s |
| 5-fold CV — mean-value baseline | R²≈0.00 |
| 5-fold CV — plain linear regression | R²=0.539 ± 0.022 |
| → RF vs. linear regression gap | **+0.42 R²** — real evidence the RF is capturing non-linear veto-threshold behavior, not just overfitting |
| Veto/should-mist classification (split from duration regression) | Precision 0.77, Recall 0.85, F1 0.81 |
| Duration regression on non-zero rows only | R²=0.965, MAE=1.32s |
| Residual mean / std | 0.16s / 1.56s (near-unbiased; slight underprediction on low-nonzero 1–4s durations — see `residuals.png`) |

### What does "5-fold CV" mean?

Instead of one train/test split (train on 80% of rows, test on the other 20%,
report one score), 5-fold cross-validation:

1. Splits the full dataset into **5 equal chunks** ("folds") — for the
   2000-row dataset, that's 5 chunks of 400 rows each.
2. Trains the model **5 separate times**. Each round uses 4 folds (1600 rows)
   to train, and the **1 remaining fold (400 rows) as the test set** — a
   different fold each round:

   ```
   Round 1: [TEST][train][train][train][train]
   Round 2: [train][TEST][train][train][train]
   Round 3: [train][train][TEST][train][train]
   Round 4: [train][train][train][TEST][train]
   Round 5: [train][train][train][train][TEST]
   ```
3. Every row gets used as test data exactly once, and as training data 4
   times — so you get **5 separate scores** instead of one.

**Why bother:** a single 80/20 split gives one number that partly depends on
luck — which rows happened to land in the test set. 5-fold CV instead gives a
**mean ± standard deviation** across 5 different splits, which tells you both
the average performance *and* how much it varies. A small std (like this
model's `R²=0.960 ± 0.020`) means the score is consistent and trustworthy,
not a fluke from one lucky split — that's also what makes the RF-vs-baseline
comparison above meaningful, since all three models were scored the exact
same way.

## Graphs
- `feature_importance.png` — **two methods side by side**: built-in (Gini)
  importance and permutation importance (mean R² drop) — included together
  because Gini importance can be biased by the multicollinearity flagged in
  the report (`total_count`/`co2_ppm`/`small_medium_count` are correlated).
  Both agree: `rh_pct` dominates, then `mature_count`/`mature_ratio`.
- `actual_vs_predicted.png` — predicted vs. true mist duration on the test
  set, now with **both** the y=x reference line and a fitted trend line
  through the actual predictions (slope/intercept annotated) to make bias
  visible, not just scatter tightness.
- `residuals.png` — **new**: residuals-vs-actual scatter + residual
  histogram, to check for systematic bias instead of eyeballing the scatter.
- `cv_comparison.png` — **new**: per-fold R²/MAE boxplots for the RF vs. the
  mean baseline vs. linear regression — the visual version of the "why RF"
  justification above.
- `duration_distribution.png` — histogram of `mist_duration_seconds` (incl.
  the ~31% zero/veto spike).
- `correlation_heatmap.png` — pairwise correlation of all dataset columns
  (now including `mature_ratio`).
- `stage_vs_duration.png` — mist duration by mushroom-stage count (3 boxplots).
- `rh_vs_duration.png` — humidity vs. mist duration, colored by
  `small_medium_count`, with a regression line.

## Run it
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python scripts/mushroom_rf_pipeline.py
```
Regenerates the synthetic dataset, retrains, re-evaluates (CV + baselines +
permutation importance), and rewrites all graphs + the model file + the
evaluation report.

Source of truth for ongoing development remains the `IIICe-SP` repo
(`origin/main` branch, `scripts/mushroom_rf_pipeline.py`); this folder is a
snapshot for the YOLOv11 codebase.
