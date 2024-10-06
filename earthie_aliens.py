### Complete Code:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import requests
from datetime import datetime, timedelta

# Set Meteomatics API credentials
username = 'nasaspacechallenge_aliens_earthie'
password = 'jDh9W2p5K4'

# Define a list of cities with lat/lon for United States cities only
locations = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'San Francisco': (37.7749, -122.4194),
    'Miami': (25.7617, -80.1918),
    'Dallas': (32.7767, -96.7970),
    'Washington DC': (38.9072, -77.0369),
    'Atlanta': (33.7490, -84.3880),
    'Boston': (42.3601, -71.0589)
}

# Function to fetch real-time weather data from Meteomatics API
def fetch_weather_data(lat, lon, days=7):
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=days)
    url = f"https://api.meteomatics.com/{start_date.isoformat()}Z--{end_date.isoformat()}Z:PT1H/t_2m:C,precip_24h:mm,wind_speed_10m:ms/{lat},{lon}/json"
    response = requests.get(url, auth=(username, password))
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Data Preprocessing and Loading Module using real-time weather data
def load_and_merge_datasets(location_name, days=7):
    lat, lon = locations.get(location_name)  # Get the lat/lon from the US cities
    weather_data = fetch_weather_data(lat, lon, days=days)  # Fetch 7 days of weather data
    if weather_data:
        # Extracting relevant fields from the weather data
        times = []
        temps = []
        precip = []
        wind_speed = []

        for entry in weather_data['data'][0]['coordinates'][0]['dates']:
            times.append(entry['date'])
            temps.append(entry['value'])

        for entry in weather_data['data'][1]['coordinates'][0]['dates']:
            precip.append(entry['value'])

        for entry in weather_data['data'][2]['coordinates'][0]['dates']:
            wind_speed.append(entry['value'])

        # Creating DataFrame
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(times),
            'temperature': temps,
            'precipitation': precip,
            'wind_speed': wind_speed
        })

        # Generating synthetic soil moisture data (replace with real soil data if available)
        soil_moisture = np.random.uniform(10, 40, size=(len(times),))  # Random soil moisture for demonstration

        # Combine into a single dataset
        merged_data = weather_df.copy()
        merged_data['soil_moisture'] = soil_moisture
        return merged_data
    else:
        print("Error fetching or processing weather data.")
        return None

# Example crop suitability data for different regions with relaxed conditions
crop_data = {
    'Corn': {'temp_range': (8, 40), 'precip_range': (100, 700), 'moisture_range': (15, 60)},
    'Wheat': {'temp_range': (0, 30), 'precip_range': (100, 500), 'moisture_range': (10, 50)},
    'Rice': {'temp_range': (10, 35), 'precip_range': (300, 1000), 'moisture_range': (30, 80)},
    'Soybeans': {'temp_range': (8, 40), 'precip_range': (150, 600), 'moisture_range': (20, 60)},
    'Barley': {'temp_range': (0, 25), 'precip_range': (80, 400), 'moisture_range': (10, 40)},
    'Sorghum': {'temp_range': (10, 45), 'precip_range': (50, 400), 'moisture_range': (5, 45)}
}

# Crop recommendation function with relaxed conditions
def recommend_crops(current_temp, current_precip, current_moisture):
    suitable_crops = []
    for crop, data in crop_data.items():
        temp_range = data['temp_range']
        precip_range = data['precip_range']
        moisture_range = data['moisture_range']

        # Require at least two conditions to be met for a recommendation
        conditions_met = 0
        if temp_range[0] <= current_temp <= temp_range[1]:
            conditions_met += 1
        if precip_range[0] <= current_precip <= precip_range[1]:
            conditions_met += 1
        if moisture_range[0] <= current_moisture <= moisture_range[1]:
            conditions_met += 1

        if conditions_met >= 2:  # Relaxing the condition to require only 2 matches
            suitable_crops.append(crop)

    return suitable_crops

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

# Loop over each US city and apply the model
for city in locations.keys():
    print(f"\nProcessing data for {city}...\n")
    
    # Load and Scale Data for each city
    data = load_and_merge_datasets(city)
    if data is not None:
        X_scaled, y = scale_data(data, 'soil_moisture')

        # Use real-time weather data to recommend crops
        current_temperature = data['temperature'].mean()  # Average temperature over the period
        current_precipitation = data['precipitation'].mean()  # Average precipitation over the period
        current_soil_moisture = y.mean()  # Average predicted soil moisture (or replace with actual data)

        print(f"Current conditions for {city}: Temp={current_temperature}, Precip={current_precipitation}, Moisture={current_soil_moisture}")
        
        recommended_crops = recommend_crops(current_temperature, current_precipitation, current_soil_moisture)
        print(f"Recommended crops for {city}: {recommended_crops}")

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
        plt.title(f"Loss for {city}")
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
        shap.summary_plot(shap_values, features=data.drop(columns=['date', 'soil_moisture']).iloc[:len(X_test_2d)], title=f"SHAP Summary for {city}")
    else:
        print(f"Data loading failed for {city}.")
