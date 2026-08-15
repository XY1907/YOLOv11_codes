"""
Mushroom Irrigation Random Forest Regressor Pipeline
Oyster mushroom (Pleurotus florida / P. pulmonarius warm-strain) automated misting system
Part of SpotShrooms / OSIP Mushroom. Predicts mist_duration_seconds from live camera
counts (YOLO growth-stage detector) + ambient sensor readings (RH/temp/CO2).

────────────────────────────────────────────────────────────────────────────
ASSUMPTIONS & LITERATURE BASIS  (read this before trusting the numbers below)
────────────────────────────────────────────────────────────────────────────
This pipeline trains and evaluates entirely on SYNTHETIC data: no real hut
sensor, camera, or actuator logs exist yet. Every metric printed/plotted here
(R2, MAE, feature importance, etc.) therefore only shows how well the RF can
recover the hand-written label formula below -- it is NOT evidence that the
formula (or the model) matches real misting needs. Treat this as "pipeline
correctness + statistical rigor" validation, not "real-world accuracy".
Per the project's design spec, the same pipeline should retrain unchanged
once real hut data lands -- that retrain is the actual accuracy test.

Where the synthetic label rules came from:
  - RH / temp bands (fruiting target ~85-95% RH, 20-28C, spawn 25-30C/70-75%,
    >30C thermal stress) are grounded in the same sources cited in
    docs/superpowers/specs/2026-06-13-humidity-temp-irrigation-timing-design.md #3:
      * IJRRR 2018 -- Temperature & RH effects on P. florida fruiting body production
      * ResearchGate -- RH & Temperature effects on Pleurotus species cultivation
      * Rhizo Funga -- P. pulmonarius (Phoenix) warm-strain fruiting 18-24C, RH 85-95%
      * Shroomability -- oyster incubation vs fruiting temperatures
      * Penn State Extension -- Bacterial Blotch Disease (drives the RH-saturation veto)
  - The RH>=92% veto is a DELIBERATE ~3pp safety margin below the project's cited
    rh_hard_max=95% (config/oyster.yaml) -- a design choice, not itself a cited number.
  - The mature-ratio (>=5%) veto is SOFTER than the sibling irrigation_timing model's
    blanket "never irrigate mature" rule. Rationale: a shed shelf typically holds
    mixed-stage mushrooms, so a small mature fraction shouldn't stop misting the
    co-located pinning/fruiting ones; a larger fraction signals "prioritise harvest".
    This is a business-logic choice, not literature-cited -- flagged for validation
    against real harvest data.
  - CO2 thresholds (800ppm baseline, elevated contribution above ~1200/1800/2000ppm)
    reflect commonly-cited oyster-cultivation guidance that elevated CO2 during
    fruiting promotes long stems/small caps, but NO specific primary source is
    captured in this file -- treat these numbers as uncited/unverified pending a
    real citation, unlike the RH/temp bands above.
  - Species label corrected to P. florida/pulmonarius (was P. ostreatus in the first
    version of this script) to match the tropical strain the rest of the project
    (config/oyster.yaml) targets.
────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.inspection import permutation_importance

np.random.seed(42)
seaborn_palette = "Greens"
REPORT_LINES = []  # collects text for mushroom_rf_evaluation_report.txt


def log(msg=""):
    """Print and also capture into the evaluation report."""
    print(msg)
    REPORT_LINES.append(str(msg))


# ─────────────────────────────────────────────────────────────
# STEP 1 — Generate Synthetic Dataset
# ─────────────────────────────────────────────────────────────
N = 2000  # bumped from 500 -> tighter CV estimates (see assumptions block: still synthetic)
log("=" * 60)
log(f"STEP 1: Generating synthetic dataset ({N} rows)")
log("=" * 60)

# Simulate grow cycle phases: 30% early, 50% mid, 20% late
cycle_phases = np.random.choice(["early", "mid", "late"], size=N, p=[0.30, 0.50, 0.20])
cycle_day = np.where(cycle_phases == "early",
                     np.random.randint(1, 6, N),
                     np.where(cycle_phases == "mid",
                              np.random.randint(5, 13, N),
                              np.random.randint(8, 21, N)))

# Hour of day (affects temp and humidity)
hour = np.random.randint(0, 24, N)
morning_mask = (hour >= 6) & (hour <= 10)
afternoon_mask = (hour >= 12) & (hour <= 16)

# Temperature: higher in afternoon, lower at night
temp_base = 22.0
temp_c = np.random.normal(temp_base, 2, N)
temp_c[afternoon_mask] += np.random.uniform(1.5, 3.0, afternoon_mask.sum())
temp_c[~afternoon_mask & ~morning_mask] -= np.random.uniform(0.5, 1.5,
    (~afternoon_mask & ~morning_mask).sum())
temp_c = np.clip(temp_c, 16, 28)

# Humidity: inversely correlated with temp, higher in morning
rh_base = 88.0
rh_pct = np.random.normal(rh_base, 6, N)
rh_pct[morning_mask] += np.random.uniform(2, 5, morning_mask.sum())
rh_pct[afternoon_mask] -= np.random.uniform(2, 6, afternoon_mask.sum())
rh_pct -= (temp_c - temp_base) * 0.8   # inverse correlation with temp
rh_pct = np.clip(rh_pct, 60, 98)

# Mushroom counts by cycle phase (Poisson distribution)
no_mushroom_sprout_count = np.where(cycle_phases == "early",
    np.random.poisson(6, N),
    np.where(cycle_phases == "mid",
             np.random.poisson(2, N),
             np.random.poisson(0.5, N))).astype(int)
no_mushroom_sprout_count = np.clip(no_mushroom_sprout_count, 0, 12)

small_medium_count = np.where(cycle_phases == "early",
    np.random.poisson(1, N),
    np.where(cycle_phases == "mid",
             np.random.poisson(8, N),
             np.random.poisson(3, N))).astype(int)
small_medium_count = np.clip(small_medium_count, 0, 15)

mature_count = np.where(cycle_day < 8,
    np.zeros(N, dtype=int),
    np.random.poisson(1.5, N)).astype(int)
mature_count = np.clip(mature_count, 0, 6)

total_count = no_mushroom_sprout_count + small_medium_count + mature_count
mature_ratio = np.divide(mature_count, total_count,
                          out=np.zeros(N, dtype=float), where=total_count > 0)

# CO2: baseline + per-mushroom contribution + time-of-day effect
co2_ppm = 800 + total_count * 50.0
co2_ppm += small_medium_count * 30.0       # active growth = more CO2
co2_ppm += mature_count * 60.0             # peak respiration at maturity
night_mask = (hour >= 22) | (hour <= 5)
co2_ppm[night_mask] += np.random.uniform(150, 400, night_mask.sum())  # less air exchange
co2_ppm += np.random.normal(0, 80, N)      # sensor noise
co2_ppm = np.clip(co2_ppm, 600, 2500)

# 5% DHT22 sensor drift rows
drift_mask = np.random.random(N) < 0.05
rh_pct[drift_mask] += np.random.uniform(-5, 5, drift_mask.sum())
temp_c[drift_mask] += np.random.uniform(-2, 2, drift_mask.sum())
rh_pct = np.clip(rh_pct, 60, 98)
temp_c = np.clip(temp_c, 16, 28)

# ── Mist duration ground truth labelling (see ASSUMPTIONS block above) ──
mist_duration = np.zeros(N)

for i in range(N):
    rh = rh_pct[i]
    temp = temp_c[i]
    co2 = co2_ppm[i]
    sm = small_medium_count[i]
    sprout = no_mushroom_sprout_count[i]
    mat = mature_count[i]
    total = total_count[i]

    # Humidity contribution (main driver)
    if rh < 70:
        base = 25
    elif rh < 78:
        base = 18
    elif rh < 84:
        base = 12
    elif rh < 88:
        base = 7
    elif rh < 92:
        base = 3
    else:
        base = 0

    # Mushroom stage contribution
    base += sm * 1.0
    base += sprout * 0.6
    base += mat * 0.0

    # CO2 contribution
    if co2 > 1800:
        base += 4
    elif co2 > 1200:
        base += 2

    # Temperature contribution
    if temp > 25:
        base += 3
    elif temp > 22:
        base += 1.5

    # Veto conditions
    mr = mat / total if total > 0 else 0
    if mr >= 0.05:
        base = 0
    if rh >= 92:
        base = 0
    if co2 > 2000:
        base = 0

    # Realistic noise
    base += np.random.normal(0, 1.5)
    mist_duration[i] = max(0, round(base, 1))

# Build dataframe
df = pd.DataFrame({
    "no_mushroom_sprout_count": no_mushroom_sprout_count,
    "small_medium_count": small_medium_count,
    "mature_count": mature_count,
    "total_count": total_count,
    "mature_ratio": np.round(mature_ratio, 4),
    "rh_pct": np.round(rh_pct, 1),
    "temp_c": np.round(temp_c, 1),
    "co2_ppm": np.round(co2_ppm, 0).astype(int),
    "mist_duration_seconds": mist_duration,
})

df.to_csv("mushroom_training_data.csv", index=False)
log(f"Saved: mushroom_training_data.csv ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Explore Dataset
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 2: Dataset Exploration")
log("=" * 60)

log(f"\nShape: {df.shape}")
log(f"\nBasic stats:\n{df.describe().round(2)}")
zero_n = (df["mist_duration_seconds"] == 0).sum()
log("\nMist duration distribution:")
log(f"  Zero (veto): {zero_n} rows ({zero_n / len(df) * 100:.1f}%)")
log(f"  Non-zero:    {len(df) - zero_n} rows")
log(f"  Mean:        {df['mist_duration_seconds'].mean():.2f}s")
log(f"  Max:         {df['mist_duration_seconds'].max():.1f}s")

log("\nCorrelation with mist_duration_seconds:")
log(df.corr(numeric_only=True)["mist_duration_seconds"].sort_values(ascending=False).round(3))

# Automated multicollinearity check among FEATURES (informs how to read importances)
FEATURES = ["no_mushroom_sprout_count", "small_medium_count", "mature_count",
            "total_count", "mature_ratio", "rh_pct", "temp_c", "co2_ppm"]
feat_corr = df[FEATURES].corr(numeric_only=True)
log("\nMulticollinearity check (|correlation| > 0.7 among features):")
flagged = []
for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        c = feat_corr.loc[a, b]
        if abs(c) > 0.7:
            flagged.append((a, b, c))
            log(f"  {a} <-> {b}: {c:.2f}")
if not flagged:
    log("  none")
else:
    log("  -> importance scores for these pairs should be read as shared/interchangeable,")
    log("     not independent -- see permutation importance in Step 6d for a cross-check.")

# ─────────────────────────────────────────────────────────────
# STEP 3-4 — Features, Target, Train/Test Split
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 3-4: Features, Target, Train/Test Split")
log("=" * 60)

X = df[FEATURES]
y = df["mist_duration_seconds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Train Model
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 5: Training RandomForestRegressor")
log("=" * 60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    oob_score=True,  # free extra validation on out-of-bag samples, no extra split needed
)
model.fit(X_train, y_train)
log("Model trained.")
log(f"Out-of-bag R^2 (bagging-internal validation, independent of the test split): "
    f"{model.oob_score_:.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Evaluate on held-out test set
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 6: Evaluation on Test Set (single 80/20 split)")
log("=" * 60)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

log(f"  MAE:  {mae:.3f} seconds")
log(f"  RMSE: {rmse:.3f} seconds")
log(f"  R2:   {r2:.3f}")
log("  NOTE: a single split on ~2000 rows can still shift a few points either way --")
log("        see Step 6a for cross-validated estimates, which are the more defensible number.")

# ─────────────────────────────────────────────────────────────
# STEP 6a — 5-fold cross-validation + baseline comparison
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 6a: 5-fold Cross-Validation vs. Baselines")
log("=" * 60)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "Mean baseline (DummyRegressor)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest (this model)": RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42),
}
cv_scores = {}  # name -> {"r2": array, "mae": array, "rmse": array}
for name, est in candidates.items():
    res = cross_validate(
        est, X, y, cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    cv_scores[name] = {
        "r2": res["test_r2"],
        "mae": -res["test_neg_mean_absolute_error"],
        "rmse": -res["test_neg_root_mean_squared_error"],
    }
    log(f"\n{name}")
    log(f"  R2   : {res['test_r2'].mean():.3f} +/- {res['test_r2'].std():.3f}")
    log(f"  MAE  : {-res['test_neg_mean_absolute_error'].mean():.3f} +/- "
        f"{res['test_neg_mean_absolute_error'].std():.3f} s")
    log(f"  RMSE : {-res['test_neg_root_mean_squared_error'].mean():.3f} +/- "
        f"{res['test_neg_root_mean_squared_error'].std():.3f} s")

rf_mean_r2 = cv_scores["Random Forest (this model)"]["r2"].mean()
lin_mean_r2 = cv_scores["Linear Regression"]["r2"].mean()
dummy_mean_r2 = cv_scores["Mean baseline (DummyRegressor)"]["r2"].mean()
log(f"\n-> RF beats the mean baseline by {rf_mean_r2 - dummy_mean_r2:+.3f} R2 "
    f"and beats plain linear regression by {rf_mean_r2 - lin_mean_r2:+.3f} R2 "
    f"(cross-validated, 5-fold). This is the actual justification for using an RF here --")
log("   the mean baseline is near 0 R2 by construction, so the interesting comparison is")
log("   RF vs. linear regression: a meaningfully positive gap means the RF is capturing")
log("   real non-linearity/interactions (e.g. the veto thresholds), not just overfitting.")

# ─────────────────────────────────────────────────────────────
# STEP 6b — Zero (veto) vs. non-zero split evaluation
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 6b: Split Evaluation — Veto Decision vs. Duration Regression")
log("=" * 60)
log("Rationale: ~%.0f%% of labels are exact zeros from hard veto rules, which are" %
    (zero_n / len(df) * 100))
log("near-trivial to predict and inflate the global R2 above. Splitting the evaluation")
log("into (a) 'should we mist at all' and (b) duration accuracy on the rows where we")
log("should is a more honest picture.")

MIST_THRESHOLD = 0.5  # seconds; predictions below this are treated as "no mist"
actual_should_mist = (y_test > 0).astype(int)
pred_should_mist = (y_pred > MIST_THRESHOLD).astype(int)

log(f"\n(a) Veto/should-mist classification (actual>0 vs. predicted>{MIST_THRESHOLD}s):")
log(f"  Precision : {precision_score(actual_should_mist, pred_should_mist, zero_division=0):.3f}")
log(f"  Recall    : {recall_score(actual_should_mist, pred_should_mist, zero_division=0):.3f}")
log(f"  Accuracy  : {accuracy_score(actual_should_mist, pred_should_mist):.3f}")
log(f"  F1        : {f1_score(actual_should_mist, pred_should_mist, zero_division=0):.3f}")

nonzero_mask = y_test > 0
if nonzero_mask.sum() > 1:
    mae_nz = mean_absolute_error(y_test[nonzero_mask], y_pred[nonzero_mask])
    rmse_nz = mean_squared_error(y_test[nonzero_mask], y_pred[nonzero_mask]) ** 0.5
    r2_nz = r2_score(y_test[nonzero_mask], y_pred[nonzero_mask])
    log(f"\n(b) Duration regression on non-zero rows only (n={int(nonzero_mask.sum())}):")
    log(f"  MAE  : {mae_nz:.3f} seconds")
    log(f"  RMSE : {rmse_nz:.3f} seconds")
    log(f"  R2   : {r2_nz:.3f}")
    log("  (Compare this R2 to the global 0.9-ish figure above -- if it's notably lower,")
    log("   the global number was partly inflated by the easy zero cluster.)")

# ─────────────────────────────────────────────────────────────
# STEP 6c — Permutation importance (cross-check vs. built-in Gini importance)
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 6c: Permutation Importance (test set, 30 repeats)")
log("=" * 60)
log("Built-in RF feature_importances_ (Gini/impurity-based) can be biased toward")
log("high-cardinality or correlated features (see Step 2 multicollinearity check).")
log("Permutation importance -- how much R2 drops when a feature is shuffled -- is a")
log("more robust, model-agnostic cross-check.")

perm = permutation_importance(model, X_test, y_test, n_repeats=30,
                               random_state=42, scoring="r2")
gini_importances = pd.Series(model.feature_importances_, index=FEATURES)
perm_importances = pd.Series(perm.importances_mean, index=FEATURES)
log("\nFeature            Gini imp.   Permutation imp. (mean R2 drop)")
for f in FEATURES:
    log(f"  {f:<24s} {gini_importances[f]:.3f}       {perm_importances[f]:.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — Graphs
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 7: Generating graphs")
log("=" * 60)

sns.set_theme(style="whitegrid")
TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 16, 13, 11

# Graph 1 — Feature Importance: Gini (built-in) vs. Permutation, side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
gini_sorted = gini_importances.sort_values()
colors = sns.color_palette("RdYlGn", len(gini_sorted))
bars = axes[0].barh(gini_sorted.index, gini_sorted.values, color=colors)
for bar, val in zip(bars, gini_sorted.values):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=TICK_SIZE)
axes[0].set_title("Built-in (Gini) Importance", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Importance Score", fontsize=LABEL_SIZE)
axes[0].tick_params(labelsize=TICK_SIZE)

perm_sorted = perm_importances.sort_values()
colors2 = sns.color_palette("RdYlGn", len(perm_sorted))
bars2 = axes[1].barh(perm_sorted.index, perm_sorted.values, color=colors2)
for bar, val in zip(bars2, perm_sorted.values):
    axes[1].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=TICK_SIZE)
axes[1].set_title("Permutation Importance (mean R2 drop)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Importance Score", fontsize=LABEL_SIZE)
axes[1].tick_params(labelsize=TICK_SIZE)

fig.suptitle("Random Forest Feature Importances — Two Methods", fontsize=TITLE_SIZE, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: feature_importance.png")

# Graph 2 — Actual vs Predicted, with BOTH the perfect-prediction line AND a fitted
# trend line through the real predictions, so systematic bias is visible (not just
# scatter tightness).
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#2e7d32", ax=ax)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
slope, intercept = np.polyfit(y_test, y_pred, 1)
fit_x = np.array([mn, mx])
ax.plot(fit_x, slope * fit_x + intercept, color="#1565c0", linewidth=1.5,
        linestyle="-", label=f"Actual fit (slope={slope:.2f}, intercept={intercept:.2f})")
ax.text(0.05, 0.90, f"MAE = {mae:.2f}s    R2 = {r2:.3f}\n5-fold CV R2 = {rf_mean_r2:.3f}",
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_title("Actual vs Predicted Misting Duration", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Actual Duration (seconds)", fontsize=LABEL_SIZE)
ax.set_ylabel("Predicted Duration (seconds)", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: actual_vs_predicted.png")

# Graph 2b (NEW) — Residual diagnostics: residuals vs actual + residual histogram
residuals = y_pred - y_test.values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_test, residuals, alpha=0.5, color="#00695c")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_title("Residuals vs. Actual Duration", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Actual Duration (seconds)", fontsize=LABEL_SIZE)
axes[0].set_ylabel("Residual = Predicted - Actual (seconds)", fontsize=LABEL_SIZE)
axes[0].tick_params(labelsize=TICK_SIZE)

sns.histplot(residuals, kde=True, color="#00695c", bins=30, ax=axes[1])
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title(f"Residual Distribution (mean={residuals.mean():.2f}s, "
                   f"std={residuals.std():.2f}s)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Residual (seconds)", fontsize=LABEL_SIZE)
axes[1].tick_params(labelsize=TICK_SIZE)

fig.suptitle("Residual Diagnostics — checking for systematic bias", fontsize=TITLE_SIZE, fontweight="bold")
plt.tight_layout()
plt.savefig("residuals.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: residuals.png")

# Graph 3 — Duration Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["mist_duration_seconds"], kde=True, color="#388e3c",
             bins=30, ax=ax, line_kws={"linewidth": 2})
ax.set_title("Distribution of Misting Duration", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Misting Duration (seconds)", fontsize=LABEL_SIZE)
ax.set_ylabel("Count", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
plt.tight_layout()
plt.savefig("duration_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: duration_distribution.png")

# Graph 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(11, 7))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=TITLE_SIZE, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: correlation_heatmap.png")

# Graph 5 — Stage vs Duration (3 side-by-side boxplots)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stage_cols = ["no_mushroom_sprout_count", "small_medium_count", "mature_count"]
stage_titles = ["No Mushroom Sprout Count", "Small/Medium Count", "Mature Count"]
palette = sns.color_palette("muted", 3)

for ax, col, title, color in zip(axes, stage_cols, stage_titles, palette):
    temp_df = df[[col, "mist_duration_seconds"]].copy()
    temp_df[col] = temp_df[col].astype(str)
    order = sorted(temp_df[col].unique(), key=lambda x: int(x))
    sns.boxplot(data=temp_df, x=col, y="mist_duration_seconds",
                order=order, color=color, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mist Duration (s)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

fig.suptitle("Misting Duration by Mushroom Stage Count",
             fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("stage_vs_duration.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: stage_vs_duration.png")

# Graph 6 — Humidity vs Misting Duration
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="rh_pct", y="mist_duration_seconds",
                hue="small_medium_count", palette="YlGn",
                alpha=0.6, ax=ax, legend="brief")
sns.regplot(data=df, x="rh_pct", y="mist_duration_seconds",
            scatter=False, ax=ax, color="red",
            line_kws={"linewidth": 1.5, "linestyle": "--"})
ax.set_title("Humidity vs Misting Duration", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Relative Humidity (%)", fontsize=LABEL_SIZE)
ax.set_ylabel("Misting Duration (seconds)", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
plt.tight_layout()
plt.savefig("rh_vs_duration.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: rh_vs_duration.png")

# Graph 7 (NEW) — Cross-validation score spread: RF vs. baselines, per fold
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names = list(candidates.keys())
r2_data = [cv_scores[n]["r2"] for n in names]
mae_data = [cv_scores[n]["mae"] for n in names]
short_names = ["Mean\nbaseline", "Linear\nRegression", "Random\nForest"]

bp1 = axes[0].boxplot(r2_data, tick_labels=short_names, patch_artist=True)
for patch, color in zip(bp1["boxes"], ["#c62828", "#f9a825", "#2e7d32"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].scatter([i + 1 + np.random.uniform(-0.05, 0.05) for i in range(len(names)) for _ in r2_data[i]],
                [v for arr in r2_data for v in arr], color="black", alpha=0.6, zorder=3, s=15)
axes[0].set_title("5-fold CV R2 by model", fontsize=13, fontweight="bold")
axes[0].set_ylabel("R2", fontsize=LABEL_SIZE)
axes[0].tick_params(labelsize=TICK_SIZE)

bp2 = axes[1].boxplot(mae_data, tick_labels=short_names, patch_artist=True)
for patch, color in zip(bp2["boxes"], ["#c62828", "#f9a825", "#2e7d32"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].scatter([i + 1 + np.random.uniform(-0.05, 0.05) for i in range(len(names)) for _ in mae_data[i]],
                [v for arr in mae_data for v in arr], color="black", alpha=0.6, zorder=3, s=15)
axes[1].set_title("5-fold CV MAE by model", fontsize=13, fontweight="bold")
axes[1].set_ylabel("MAE (seconds)", fontsize=LABEL_SIZE)
axes[1].tick_params(labelsize=TICK_SIZE)

fig.suptitle("Cross-Validated Performance — Random Forest vs. Baselines",
             fontsize=TITLE_SIZE, fontweight="bold")
plt.tight_layout()
plt.savefig("cv_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
log("Saved: cv_comparison.png")

# ─────────────────────────────────────────────────────────────
# STEP 8 — Save Model
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 8: Saving model")
log("=" * 60)

joblib.dump(model, "mushroom_rf_model.pkl")
log("Saved: mushroom_rf_model.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 9 — Predict Function
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 9: predict() function")
log("=" * 60)

def predict(no_mushroom_sprout_count, small_medium_count, mature_count,
            total_count, rh_pct, temp_c, co2_ppm):
    # Veto: no mushrooms detected
    if total_count == 0:
        return 0, "No mushrooms detected"

    mature_ratio_ = mature_count / total_count

    # Veto: mature ratio >= 5% — alert owner, do not mist
    if mature_ratio_ >= 0.05:
        return 0, "ALERT: Mature ratio reached 5% — notify owner, skip cycle"

    # Veto: humidity already saturated
    if rh_pct >= 92:
        return 0, "VETO: Humidity already saturated"

    # Veto: CO2 too high — ventilate first
    if co2_ppm > 2000:
        return 0, "VETO: CO2 too high — ventilate first"

    # RF prediction (feature order must match FEATURES / training columns)
    duration = model.predict([[no_mushroom_sprout_count, small_medium_count,
                               mature_count, total_count, mature_ratio_,
                               rh_pct, temp_c, co2_ppm]])[0]
    duration = max(0, round(duration, 1))
    return duration, "OK"

# ─────────────────────────────────────────────────────────────
# STEP 10 — Test Scenarios
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 10: Testing predict() with 5 scenarios")
log("=" * 60)

scenarios = [
    {
        "name": "A — Early cycle, very dry",
        "args": (8, 2, 0, 10, 68, 23, 900),
        "expected": "Long misting duration",
    },
    {
        "name": "B — Mid cycle, slightly dry",
        "args": (2, 10, 0, 12, 82, 21, 1400),
        "expected": "Moderate misting duration",
    },
    {
        "name": "C — Mature veto triggered",
        "args": (0, 3, 5, 8, 85, 20, 1800),
        "expected": "0 — mature ratio 62.5%, notify owner",
    },
    {
        "name": "D — Humidity saturated veto",
        "args": (4, 6, 0, 10, 94, 19, 800),
        "expected": "0 — already saturated",
    },
    {
        "name": "E — Mixed, mature under 5%",
        "args": (1, 18, 1, 20, 79, 22, 1100),
        "expected": "0 — mature ratio exactly 5%, veto triggered, notify owner",
    },
]

for s in scenarios:
    duration, reason = predict(*s["args"])
    log(f"\n  Scenario {s['name']}")
    log(f"    Result:   {duration}s — {reason}")
    log(f"    Expected: {s['expected']}")

# ─────────────────────────────────────────────────────────────
# STEP 11 — Write consolidated justification / evaluation report
# ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 11: Writing evaluation report")
log("=" * 60)

with open("mushroom_rf_evaluation_report.txt", "w", encoding="utf-8") as fh:
    fh.write(__doc__.strip() + "\n\n")
    fh.write("\n".join(str(line) for line in REPORT_LINES))
    fh.write("\n")
log("Saved: mushroom_rf_evaluation_report.txt")

log("\n" + "=" * 60)
log("Pipeline complete.")
log("=" * 60)
