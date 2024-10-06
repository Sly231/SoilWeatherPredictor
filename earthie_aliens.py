# -*- coding: utf-8 -*-
"""Earthie-Aliens.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HDWEJrrwKtq34zWldngZ_M4HWMnMp_uu
"""

!pip install shap

!pip install shap tensorflow

!pip install tensorflow shap

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

# Data Preprocessing and Loading Module
def load_and_merge_datasets():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    soil_data = pd.DataFrame({
        'date': dates,
        'soil_moisture': np.random.uniform(10, 40, size=(100,)),  # random soil moisture values
    })

    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.uniform(20, 35, size=(100,)),  # random temperatures
        'humidity': np.random.uniform(50, 90, size=(100,)),  # random humidity
    })

    precip_data = pd.DataFrame({
        'date': dates,
        'precipitation': np.random.uniform(0, 10, size=(100,)),  # random precipitation values
    })

    merged_data = pd.merge(pd.merge(soil_data, weather_data, on='date'), precip_data, on='date')
    return merged_data

# Data Scaling Function
def scale_data(data, target_column):
    X = data.drop(columns=['date', target_column])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Model Building Function
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Set input shape properly
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load and Scale Data
data = load_and_merge_datasets()
X_scaled, y = scale_data(data, 'soil_moisture')

# Reshape for LSTM (samples, timesteps, features)
X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_3d, y, test_size=0.2, random_state=42)

# Build and Train Model
model = build_model((X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Plotting the training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# SHAP Explanation
# Reshape X_test back to 2D for SHAP
X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[2])

# Define a wrapper for model prediction that reshapes the input to 3D using TensorFlow
@tf.function
def model_predict_3d(X):
    X_reshaped = tf.reshape(X, (tf.shape(X)[0], 1, tf.shape(X)[1]))  # Reshape to 3D using TensorFlow
    return model(X_reshaped)

# Create SHAP KernelExplainer with reshaped data
explainer = shap.KernelExplainer(lambda x: model_predict_3d(x).numpy(), X_test_2d)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_2d)

# SHAP Summary Plot
shap.summary_plot(shap_values, features=data.drop(columns=['date', 'soil_moisture']).iloc[:len(X_test_2d)])