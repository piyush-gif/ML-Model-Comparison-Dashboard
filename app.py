import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv('data/winequality-red.csv', sep=';')
# print(df.head())

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# Features (all columns except 'quality')
X = df.drop('quality', axis=1)
# Target (the column to predict)
y = df['quality']

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}") # Train shape: (1279, 11), Test shape: (320, 11)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.2f}")

# Train the Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Make predictions with Decision Tree
y_pred_dt = dt.predict(X_test)

# Evaluate the Decision Tree model
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"\nDecision Tree Results:")
print(f"RMSE: {rmse_dt:.2f}")
print(f"MAE: {mae_dt:.2f}")
print(f"R^2: {r2_dt:.2f}")

# Visualize the metrics for Linear Regression
metrics = ['RMSE', 'MAE', 'R^2']
values = [rmse, mae, r2]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Linear Regression Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, max(values) + 0.5)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.show()


# Visualize the metrics for Decision Tree
metrics = ['RMSE', 'MAE', 'R^2']
values = [rmse_dt, mae_dt, r2_dt]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Decision Tree Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, max(values) + 0.5)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.show()












