import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import os

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print("California Housing Dataset Overview")
print("----------------------------------")
print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target variable: {y.name}")
print("\nTarget variable statistics:")
print(y.describe())

# =========================
# Plot 1: Price Distribution
# =========================
plt.figure(figsize=(6, 4))
plt.hist(y, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.title('Distribution of Housing Prices')
plt.tight_layout()
plt.savefig("assets/price_distribution.png")
plt.close()

# =========================
# Plot 2: Feature Correlation
# =========================
correlations = X.corrwith(y).sort_values(ascending=False)

plt.figure(figsize=(6, 4))
plt.barh(correlations.index, correlations.values)
plt.xlabel('Correlation with MedHouseVal')
plt.title('Feature Correlation with House Price')
plt.tight_layout()
plt.savefig("assets/feature_correlation.png")
plt.close()

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Average House Price: ${y.mean() * 100000:.2f}")
print(f"RMSE in Dollars: ${rmse * 100000:.2f}")

print("\nPlots saved successfully in 'assets/' folder.")
