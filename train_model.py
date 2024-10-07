
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the CSV data into a DataFrame
df = pd.read_csv('earthquake_data.csv')

# Drop irrelevant columns
df = df.drop(['id', 'place', 'type', 'status', 'locationSource', 'magSource'], axis=1)
df.dropna(inplace=True)

# Convert categorical data using one-hot encoding
df = pd.get_dummies(df, columns=['magType', 'net'])

# Convert date columns to datetime format
df['time'] = pd.to_datetime(df['time'])
df['updated'] = pd.to_datetime(df['updated'])

# Extract useful features from datetime columns
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['time_diff'] = (df['updated'] - df['time']).dt.total_seconds()

# Drop the original datetime columns
df = df.drop(['time', 'updated'], axis=1)

# Define features (X) and target (y)
X = df.drop('mag', axis=1)
y = df['mag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

import joblib

# Save the trained model to a file
model_filename = 'earthquake_model.pkl'
joblib.dump(model, model_filename)
print(f'Model saved to {model_filename}')

