import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
file_path = "Sum of avg_weather_index and Sum of last_mile_congestion_index by avg_dispatch_delay_min.csv"
df = pd.read_csv(file_path)

# Show columns
print("Columns:", df.columns)

# Rename columns (Power BI names are messy)
df.columns = [
    "avg_dispatch_delay_min",
    "avg_weather_index",
    "last_mile_congestion_index"
]

# Features & Target
X = df[["avg_dispatch_delay_min", "avg_weather_index"]]
y = df["last_mile_congestion_index"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Congestion Index")
plt.ylabel("Predicted Congestion Index")
plt.title("Actual vs Predicted")
plt.show()
