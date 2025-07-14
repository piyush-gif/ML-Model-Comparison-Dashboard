import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
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














