# ==============================
# PROPER ML PIPELINE FOR level1.csv
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1️⃣ Load Dataset
df = pd.read_csv("level1.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# 2️⃣ Basic Cleaning
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

# 3️⃣ Auto-select Target (Last Column)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nTarget Column:", df.columns[-1])

# 4️⃣ Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6️⃣ Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Prediction
pred = model.predict(X_test)

# 8️⃣ Evaluation
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred, squared=False))

# 9️⃣ Feature Importance
importance = model.feature_importances_
features = df.columns[:-1]

plt.figure()
plt.bar(features, importance)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()

# 🔟 Save Model
joblib.dump(model, "level1_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel & Scaler Saved Successfully!")
