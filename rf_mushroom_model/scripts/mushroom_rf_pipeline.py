"""
Mushroom Irrigation Random Forest Regressor Pipeline
Oyster mushroom (Pleurotus ostreatus) automated misting system
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
seaborn_palette = "Greens"

# ─────────────────────────────────────────────────────────────
# STEP 1 — Generate Synthetic Dataset
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating synthetic dataset (500 rows)")
print("=" * 60)

N = 500

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

# ── Mist duration ground truth labelling ──
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
    mature_ratio = mat / total if total > 0 else 0
    if mature_ratio >= 0.05:
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
    "rh_pct": np.round(rh_pct, 1),
    "temp_c": np.round(temp_c, 1),
    "co2_ppm": np.round(co2_ppm, 0).astype(int),
    "mist_duration_seconds": mist_duration,
})

df.to_csv("mushroom_training_data.csv", index=False)
print(f"Saved: mushroom_training_data.csv ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Explore Dataset
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Dataset Exploration")
print("=" * 60)

print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe().round(2)}")
print(f"\nMist duration distribution:")
print(f"  Zero (veto): {(df['mist_duration_seconds'] == 0).sum()} rows "
      f"({(df['mist_duration_seconds'] == 0).mean()*100:.1f}%)")
print(f"  Non-zero:    {(df['mist_duration_seconds'] > 0).sum()} rows")
print(f"  Mean:        {df['mist_duration_seconds'].mean():.2f}s")
print(f"  Max:         {df['mist_duration_seconds'].max():.1f}s")

print(f"\nCorrelation with mist_duration_seconds:")
print(df.corr(numeric_only=True)["mist_duration_seconds"].sort_values(ascending=False).round(3))

# ─────────────────────────────────────────────────────────────
# STEP 3-4 — Features, Target, Train/Test Split
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3-4: Features, Target, Train/Test Split")
print("=" * 60)

FEATURES = ["no_mushroom_sprout_count", "small_medium_count", "mature_count",
            "total_count", "rh_pct", "temp_c", "co2_ppm"]

X = df[FEATURES]
y = df["mist_duration_seconds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Train Model
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training RandomForestRegressor")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained.")

# ─────────────────────────────────────────────────────────────
# STEP 6 — Evaluate
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Evaluation on Test Set")
print("=" * 60)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

print(f"  MAE:  {mae:.3f} seconds")
print(f"  RMSE: {rmse:.3f} seconds")
print(f"  R2:   {r2:.3f}")

print("\nActual vs Predicted (10 sample rows):")
sample = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10].round(1)})
print(sample.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 7 — Graphs
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Generating graphs")
print("=" * 60)

sns.set_theme(style="whitegrid")
TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 16, 13, 11

# Graph 1 — Feature Importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("RdYlGn", len(importances))
bars = ax.barh(importances.index, importances.values, color=colors)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=TICK_SIZE)
ax.set_title("Random Forest Feature Importances", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=LABEL_SIZE)
ax.set_ylabel("Feature Name", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# Graph 2 — Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#2e7d32", ax=ax)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.text(0.05, 0.92, f"MAE = {mae:.2f}s    R2 = {r2:.3f}",
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
print("Saved: actual_vs_predicted.png")

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
print("Saved: duration_distribution.png")

# Graph 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
corr = df.corr(numeric_only=True)
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, mask=mask, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=TITLE_SIZE, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

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
print("Saved: stage_vs_duration.png")

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
print("Saved: rh_vs_duration.png")

# ─────────────────────────────────────────────────────────────
# STEP 8 — Save Model
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving model")
print("=" * 60)

joblib.dump(model, "mushroom_rf_model.pkl")
print("Saved: mushroom_rf_model.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 9 — Predict Function
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: predict() function")
print("=" * 60)

def predict(no_mushroom_sprout_count, small_medium_count, mature_count,
            total_count, rh_pct, temp_c, co2_ppm):
    # Veto: no mushrooms detected
    if total_count == 0:
        return 0, "No mushrooms detected"

    mature_ratio = mature_count / total_count

    # Veto: mature ratio >= 5% — alert owner, do not mist
    if mature_ratio >= 0.05:
        return 0, "ALERT: Mature ratio reached 5% — notify owner, skip cycle"

    # Veto: humidity already saturated
    if rh_pct >= 92:
        return 0, "VETO: Humidity already saturated"

    # Veto: CO2 too high — ventilate first
    if co2_ppm > 2000:
        return 0, "VETO: CO2 too high — ventilate first"

    # RF prediction
    duration = model.predict([[no_mushroom_sprout_count, small_medium_count,
                               mature_count, total_count,
                               rh_pct, temp_c, co2_ppm]])[0]
    duration = max(0, round(duration, 1))
    return duration, "OK"

# ─────────────────────────────────────────────────────────────
# STEP 10 — Test Scenarios
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Testing predict() with 5 scenarios")
print("=" * 60)

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
    print(f"\n  Scenario {s['name']}")
    print(f"    Result:   {duration}s — {reason}")
    print(f"    Expected: {s['expected']}")

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)
