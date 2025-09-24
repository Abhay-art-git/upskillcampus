#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("C:/Users/likitha/Documents/PDS_VS/apy.csv")  # Ensure correct filename

# Display first few rows
print(data.head())

# Check column names and data types
print(data.info())

# Get statistical summary
print(data.describe())





from sklearn.preprocessing import LabelEncoder

# Encoding categorical features
le = LabelEncoder()
data['State'] = le.fit_transform(data['State'])
data['Crop'] = le.fit_transform(data['Crop'])
data['Season'] = le.fit_transform(data['Season'])





from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


X = data[['State', 'Crop', 'Season', 'Area']]
y = data['Production']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





print(X_train.shape, y_train.shape)





import numpy as np

# Check for missing values in X_train
print("Missing values in X_train:", np.isnan(X_train).sum())

# Check for missing values in y_train
print("Missing values in y_train:", np.isnan(y_train).sum())





# Replace missing values in X_train with the mean of each feature
X_train = np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)

# Replace missing values in y_train with the mean
y_train = np.where(np.isnan(y_train), np.nanmean(y_train), y_train)





print("Missing values in y_train:", np.isnan(y_train).sum())





model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)





y_pred = model.predict(X_test)





print(f"Missing values in y_test: {np.isnan(y_test).sum()}")
print(f"Missing values in y_pred: {np.isnan(y_pred).sum()}")





y_test = np.nan_to_num(y_test)  # Replace NaN with 0





# Evaluating the model
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-Squared Score: {r2_score(y_test, y_pred)}")





import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Crop Production")
plt.show()







