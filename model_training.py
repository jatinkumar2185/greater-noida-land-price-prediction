"""
=============================================================
model_training.py  — Greater Noida Land Price Prediction
B.Tech Major Project | ML Model Training Script
=============================================================
WHAT THIS SCRIPT DOES:
  1. Loads dataset.csv
  2. Performs Exploratory Data Analysis (EDA)
  3. Encodes categorical features (Label + One-Hot)
  4. Trains two models: Linear Regression & Random Forest
  5. Compares performance using MAE, MSE, R² Score
  6. Saves the best model as model.pkl
  7. Generates feature importance & price distribution graphs
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
GRAPHS_DIR  = os.path.join(BASE_DIR, "static", "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  GREATER NOIDA LAND PRICE PREDICTION — MODEL TRAINING")
print("="*55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# ──────────────────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────
print("\n[2] Basic Statistics:")
print(df.describe().to_string())
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── GRAPH 1: Price Distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Greater Noida — Land Price Distribution", fontsize=14, fontweight='bold')

axes[0].hist(df['Price_INR'] / 1e6, bins=40, color='#2E86AB', edgecolor='white', linewidth=0.5)
axes[0].set_title("Price Distribution (₹ in Lakhs)")
axes[0].set_xlabel("Price (₹ Lakhs)")
axes[0].set_ylabel("Number of Plots")
axes[0].grid(axis='y', alpha=0.3)

# Log-scale histogram (more informative)
axes[1].hist(np.log10(df['Price_INR']), bins=40, color='#A23B72', edgecolor='white', linewidth=0.5)
axes[1].set_title("Log-Scale Price Distribution")
axes[1].set_xlabel("log₁₀(Price in ₹)")
axes[1].set_ylabel("Frequency")
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "price_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n[2a] Graph saved: price_distribution.png")

# ── GRAPH 2: Sector-wise Average Price ────────────────────
avg_price = df.groupby('Sector')['Price_INR'].mean().sort_values(ascending=False) / 1e6
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(avg_price.index, avg_price.values, color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(avg_price))))
ax.set_title("Average Land Price by Sector (₹ Lakhs)", fontsize=13, fontweight='bold')
ax.set_xlabel("Average Price (₹ Lakhs)")
for bar, val in zip(bars, avg_price.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'₹{val:.1f}L', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "sector_avg_price.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[2b] Graph saved: sector_avg_price.png")

# ──────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING & ENCODING
# ──────────────────────────────────────────────────────────
print("\n[3] Encoding categorical features...")

df_model = df.copy()

# Label-encode binary Yes/No columns → 1/0
binary_cols = ['Corner_Plot', 'Nearby_School', 'Nearby_Hospital', 'Commercial_Nearby']
for col in binary_cols:
    df_model[col] = (df_model[col] == 'Yes').astype(int)
    print(f"   {col}: Yes→1, No→0")

# One-Hot encode 'Facing' (North/South/East/West)
df_model = pd.get_dummies(df_model, columns=['Facing'], prefix='Facing', drop_first=True)
print("   Facing: One-Hot encoded")

# Label-encode 'Sector' (ordinal by base price for better tree splits)
sector_order = [
    "Ecotech 1","Techzone IV","Sector 3","Sector 2","Sector 1",
    "Delta 2","Delta 1","Gamma 2","Gamma 1","Zeta 2","Zeta 1",
    "Knowledge Park 2","Knowledge Park 1","Omicron 2","Beta 2",
    "Omicron 1","Beta 1","Alpha 2","Alpha 1","Pari Chowk"
]
sector_map = {s: i for i, s in enumerate(sector_order)}
df_model['Sector_Code'] = df_model['Sector'].map(sector_map).fillna(10).astype(int)
df_model.drop(columns=['Sector'], inplace=True)
print("   Sector: Ordinal label encoded")

# ── Define Features (X) and Target (y) ────────────────────
TARGET = 'Price_INR'
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

# Save feature columns for Flask app (must match at prediction time)
feature_cols = list(X.columns)
with open(os.path.join(BASE_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)
print(f"\n   Feature columns saved: {feature_cols}")

# Save sector_map for Flask
with open(os.path.join(BASE_DIR, "sector_map.json"), "w") as f:
    json.dump(sector_map, f)

# ──────────────────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[4] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ──────────────────────────────────────────────────────────
# STEP 5: TRAIN MODELS
# ──────────────────────────────────────────────────────────

# --- MODEL A: Linear Regression ---
print("\n[5a] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
mse_lr  = mean_squared_error(y_test, y_pred_lr)
r2_lr   = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f"   MAE  : ₹{mae_lr:,.0f}")
print(f"   RMSE : ₹{rmse_lr:,.0f}")
print(f"   R²   : {r2_lr:.4f}")

# --- MODEL B: Random Forest ---
print("\n[5b] Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
mse_rf  = mean_squared_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"   MAE  : ₹{mae_rf:,.0f}")
print(f"   RMSE : ₹{rmse_rf:,.0f}")
print(f"   R²   : {r2_rf:.4f}")

# ──────────────────────────────────────────────────────────
# STEP 6: MODEL COMPARISON
# ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
print(f"{'Metric':<20} {'Linear Reg':>15} {'Random Forest':>15}")
print("-"*55)
print(f"{'MAE (₹)':<20} {mae_lr:>15,.0f} {mae_rf:>15,.0f}")
print(f"{'RMSE (₹)':<20} {rmse_lr:>15,.0f} {rmse_rf:>15,.0f}")
print(f"{'R² Score':<20} {r2_lr:>15.4f} {r2_rf:>15.4f}")
print("="*55)

# ── GRAPH 3: Model Comparison Bar Chart ───────────────────
metrics = ['MAE', 'RMSE', 'R² (×10⁷)']
lr_vals = [mae_lr/1e6,  rmse_lr/1e6,  r2_lr*10]
rf_vals = [mae_rf/1e6,  rmse_rf/1e6,  r2_rf*10]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, lr_vals, w, label='Linear Regression', color='#E76F51')
ax.bar(x + w/2, rf_vals, w, label='Random Forest',     color='#2A9D8F')
ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score (MAE/RMSE in ₹ Lakhs, R²×10)")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n[6] Graph saved: model_comparison.png")

# ──────────────────────────────────────────────────────────
# STEP 7: FEATURE IMPORTANCE (Random Forest)
# ──────────────────────────────────────────────────────────
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors)
ax.set_title("Feature Importance — Random Forest Model", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
for i, v in enumerate(feat_imp):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[7] Graph saved: feature_importance.png")

# ── GRAPH 5: Actual vs Predicted (Random Forest) ──────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test/1e6, y_pred_rf/1e6, alpha=0.4, color='#457B9D', edgecolors='none', s=30)
max_val = max(y_test.max(), y_pred_rf.max()) / 1e6
ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel("Actual Price (₹ Lakhs)", fontsize=11)
ax.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=11)
ax.set_title("Actual vs Predicted — Random Forest", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[7b] Graph saved: actual_vs_predicted.png")

# ──────────────────────────────────────────────────────────
# STEP 8: SAVE BEST MODEL
# ──────────────────────────────────────────────────────────
best_model = rf if r2_rf >= r2_lr else lr
best_name  = "Random Forest" if r2_rf >= r2_lr else "Linear Regression"
print(f"\n[8] Best model: {best_name} (R²={max(r2_rf,r2_lr):.4f})")

joblib.dump(best_model, MODEL_PATH)
print(f"    Saved to: {MODEL_PATH}")

# Save metrics for display in app
metrics_dict = {
    "lr":  {"mae": round(mae_lr), "rmse": round(rmse_lr), "r2": round(r2_lr, 4)},
    "rf":  {"mae": round(mae_rf), "rmse": round(rmse_rf), "r2": round(r2_rf, 4)},
    "best": best_name
}
with open(os.path.join(BASE_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_dict, f)

print("\n✅ Training complete! All artifacts saved.")
print("   Run:  python app.py  to start the web app")
