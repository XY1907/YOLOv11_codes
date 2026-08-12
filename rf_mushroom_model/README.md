# Mushroom Irrigation RF Regressor (mist-duration model)

Part of **SpotShrooms / OSIP Mushroom**. A `RandomForestRegressor` that predicts
**how long to run the misting/irrigation system (seconds)**, given live camera
counts (from the YOLO detector) and ambient sensor readings. Pulled over from
the `IIICe-SP` repo's `origin/main` branch (`scripts/mushroom_rf_pipeline.py`,
commit `6a943f4` + the `>= 5%` veto-threshold fix in `549e743`).

## Contents
- `scripts/mushroom_rf_pipeline.py` — full pipeline: synthetic data generation,
  EDA, train/test split, training, evaluation, 6 graphs, model save, and the
  `predict()` function with veto logic.
- `mushroom_rf_model.pkl` — trained model
  (`RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
  random_state=42)`).
- `mushroom_training_data.csv` — the 500-row synthetic training set
  (`np.random.seed(42)`).
- `graphs/` — evaluation plots (see below).

## Inputs (features)
| Feature | Meaning |
|---|---|
| `no_mushroom_sprout_count` | count of un-sprouted spots (camera) |
| `small_medium_count` | count of small/medium mushrooms (camera) |
| `mature_count` | count of mature mushrooms (camera) |
| `total_count` | sum of the three counts |
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

Otherwise, the RF model predicts the misting duration from all 7 features.

## Held-out evaluation (test_size=0.2, random_state=42)
| Metric | Value |
|---|---|
| MAE | 1.77 s |
| RMSE | 3.07 s |
| R² | 0.920 |

## Graphs
- `feature_importance.png` — RandomForest feature importances across all 7 inputs.
- `actual_vs_predicted.png` — predicted vs. true mist duration on the test set.
- `duration_distribution.png` — histogram of `mist_duration_seconds` (incl. the
  zero/veto spike).
- `correlation_heatmap.png` — pairwise correlation of all dataset columns.
- `stage_vs_duration.png` — mist duration by mushroom-stage count (3 boxplots).
- `rh_vs_duration.png` — humidity vs. mist duration, colored by
  `small_medium_count`, with a regression line.

## Run it
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python scripts/mushroom_rf_pipeline.py
```
Regenerates the synthetic dataset, retrains, re-evaluates, and rewrites all
graphs + the model file.

Source of truth for ongoing development remains the `IIICe-SP` repo
(`origin/main` branch, `scripts/mushroom_rf_pipeline.py`); this folder is a
snapshot for the YOLOv11 codebase.
