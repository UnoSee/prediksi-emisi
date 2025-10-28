# ==========================================================
# 🌍 GLOBAL CARBON INTENSITY FORECASTING MODEL (LightGBM)
# ==========================================================

import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1️⃣ Load datasets ===
df_carbon = pd.read_csv("carbon-intensity-electricity.csv")
df_elec = pd.read_csv("elec-fossil-nuclear-renewables.csv")
df_energy_use = pd.read_csv("per-capita-energy-use.csv")

# === 2️⃣ Rename columns for consistency ===
df_carbon = df_carbon.rename(columns={
    "Carbon intensity of electricity - gCO2/kWh": "carbon_intensity"
})
df_elec = df_elec.rename(columns={
    "Electricity from fossil fuels - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "fossil_twh",
    "Electricity from renewables - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "renewables_twh",
    "Electricity from nuclear - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "nuclear_twh",
})
df_energy_use = df_energy_use.rename(columns={
    "Primary energy consumption per capita (kWh/person)": "energy_use_per_capita"
})

# === 3️⃣ Merge datasets ===
df_merged = pd.merge(df_carbon, df_elec, on=["Entity", "Code", "Year"], how="outer")
df_merged = pd.merge(df_merged, df_energy_use, on=["Entity", "Code", "Year"], how="outer")

# Drop rows with missing target
df_merged = df_merged.dropna(subset=["carbon_intensity"])

# Fill missing numerical values with 0
df_merged = df_merged.fillna(0)

# === 4️⃣ Temporal train-test split ===
train = df_merged[df_merged["Year"] < 2020]
test  = df_merged[df_merged["Year"] >= 2020]

print(f"Training on years {train['Year'].min()}–{train['Year'].max()} ({len(train)} samples)")
print(f"Testing  on years {test['Year'].min()}–{test['Year'].max()} ({len(test)} samples)")

# === 5️⃣ Define features and target ===
features = [
    "Year",
    "fossil_twh",
    "renewables_twh",
    "nuclear_twh",
    "energy_use_per_capita",
    "Entity"
]
target = "carbon_intensity"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Convert categorical features
X_train["Entity"] = X_train["Entity"].astype("category")
X_test["Entity"] = X_test["Entity"].astype("category")

# === 6️⃣ Train LightGBM model ===
params = {
    "objective": "regression_l1",
    "metric": "l1",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

print("\n🚀 Training LightGBM model (real-world forecasting setup)...")
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

# === 7️⃣ Forecast (predict future years) ===
y_pred = model.predict(X_test)

# === 8️⃣ Evaluate model ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n🔍 Model Forecasting Performance (2020–2022):")
print(f"MAE  = {mae:.2f} gCO₂/kWh")
print(f"RMSE = {rmse:.2f} gCO₂/kWh")
print(f"R²   = {r2:.3f}")

# === 9️⃣ Save model for reuse ===
model.booster_.save_model("model_lightgbm_forecast.txt")
print("✅ Model saved as model_lightgbm_forecast.txt")

# === 🔟 Improved Visualization: Global Average (Clean Paper Version) ===

# Add predicted column to test data
test_with_pred = test.copy()
test_with_pred["predicted"] = y_pred

# Aggregate global average observed vs predicted by year
agg = test_with_pred.groupby("Year")[["carbon_intensity", "predicted"]].mean().reset_index()

# Plot clean trend
plt.figure(figsize=(8,5))
plt.plot(agg["Year"], agg["carbon_intensity"], marker="o", linewidth=2, label="Observed (Global Avg)")
plt.plot(agg["Year"], agg["predicted"], marker="x", linestyle="--", linewidth=2, label="Predicted (Global Avg)")
plt.title("Global Average Carbon Intensity (Observed vs Predicted)", fontsize=13)
plt.xlabel("Year")
plt.ylabel("Carbon Intensity (gCO₂/kWh)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
forecast_results = agg.copy()
forecast_results.to_csv("forecast_results.csv", index=False)
print("📊 Forecast results saved as forecast_results.csv")