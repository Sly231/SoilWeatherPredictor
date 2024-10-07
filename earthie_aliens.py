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
import plotly.graph_objects as go

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

# Define crop data for recommendations (example ranges)
crop_data = {
    'Corn': {'temp_range': (8, 40), 'precip_range': (100, 700), 'moisture_range': (15, 60)},
    'Wheat': {'temp_range': (0, 30), 'precip_range': (100, 500), 'moisture_range': (10, 50)},
    'Rice': {'temp_range': (10, 35), 'precip_range': (300, 1000), 'moisture_range': (30, 80)},
    'Soybeans': {'temp_range': (8, 40), 'precip_range': (150, 600), 'moisture_range': (20, 60)},
    'Barley': {'temp_range': (0, 25), 'precip_range': (80, 400), 'moisture_range': (10, 40)},
    'Sorghum': {'temp_range': (10, 45), 'precip_range': (50, 400), 'moisture_range': (5, 45)}
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
    lat, lon = locations.get(location_name)
    weather_data = fetch_weather_data(lat, lon, days=days)
    if weather_data:
        times, temps, precip, wind_speed = [], [], [], []

        for entry in weather_data['data'][0]['coordinates'][0]['dates']:
            times.append(entry['date'])
            temps.append(entry['value'])

        for entry in weather_data['data'][1]['coordinates'][0]['dates']:
            precip.append(entry['value'])

        for entry in weather_data['data'][2]['coordinates'][0]['dates']:
            wind_speed.append(entry['value'])

        # Create DataFrame
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(times),
            'temperature': temps,
            'precipitation': precip,
            'wind_speed': wind_speed
        })

        # Generate synthetic soil moisture data (for demonstration)
        soil_moisture = np.random.uniform(10, 40, size=(len(times),))
        merged_data = weather_df.copy()
        merged_data['soil_moisture'] = soil_moisture

        # Weather Plot Visualization
        visualize_weather_forecast(weather_df, location_name)

        return merged_data
    else:
        print("Error fetching or processing weather data.")
        return None

# Visualize the weather forecast
def visualize_weather_forecast(weather_df, location_name):
    fig = go.Figure()

    # Add temperature trace
    fig.add_trace(go.Scatter(
        x=weather_df['date'], y=weather_df['temperature'],
        mode='lines+markers', name='Temperature (°C)',
        marker=dict(color='red'),
        hoverinfo='text',
        text=[f"Temperature: {temp} °C" for temp in weather_df['temperature']]
    ))

    # Add precipitation trace
    fig.add_trace(go.Scatter(
        x=weather_df['date'], y=weather_df['precipitation'],
        mode='lines+markers', name='Precipitation (mm)',
        marker=dict(color='blue'),
        hoverinfo='text',
        text=[f"Precipitation: {prec} mm" for prec in weather_df['precipitation']]
    ))

    # Add wind speed trace
    fig.add_trace(go.Scatter(
        x=weather_df['date'], y=weather_df['wind_speed'],
        mode='lines+markers', name='Wind Speed (m/s)',
        marker=dict(color='green'),
        hoverinfo='text',
        text=[f"Wind Speed: {ws} m/s" for ws in weather_df['wind_speed']]
    ))

    fig.update_layout(
        title=f"Weather Forecast for {location_name}",
        xaxis_title='Date and Time',
        yaxis_title='Values',
        template='plotly_white',
        showlegend=True
    )

    fig.show()

# Function to recommend crops with detailed logging
def recommend_crops(current_temp, current_precip, current_moisture):
    suitable_crops = []
    print(f"\nChecking conditions: Temp={current_temp}, Precip={current_precip}, Moisture={current_moisture}\n")
    
    for crop, data in crop_data.items():
        temp_min = data['temp_range'][0] - 5
        temp_max = data['temp_range'][1] + 5
        precip_min = data['precip_range'][0] - 50
        precip_max = data['precip_range'][1] + 50
        moisture_min = data['moisture_range'][0] - 3
        moisture_max = data['moisture_range'][1] + 3

        print(f"Checking {crop}: Temp Range=({temp_min}, {temp_max}), Precip Range=({precip_min}, {precip_max}), Moisture Range=({moisture_min}, {moisture_max})")

        if (temp_min <= current_temp <= temp_max and
            precip_min <= current_precip <= precip_max and
            moisture_min <= current_moisture <= moisture_max):
            suitable_crops.append(crop)

    return suitable_crops

# Loop over each US city and apply the model
for city in locations.keys():
    print(f"\nProcessing data for {city}...\n")

    # Load and Scale Data for each city
    data = load_and_merge_datasets(city)
    if data is not None:
        current_temperature = data['temperature'].mean()
        current_precipitation = data['precipitation'].mean()
        current_soil_moisture = data['soil_moisture'].mean()

        print(f"\nCurrent conditions for {city}: Temp={current_temperature}, Precip={current_precipitation}, Moisture={current_soil_moisture}\n")

        # Recommend crops based on real-time weather data
        recommended_crops = recommend_crops(current_temperature, current_precipitation, current_soil_moisture)

        # Visualize the recommended crops
        if recommended_crops:
            print(f"Recommended crops for {city}: {', '.join(recommended_crops)}\n")
        else:
            print(f"No suitable crops found for {city} under current conditions.\n")
    else:
        print(f"Data loading failed for {city}.")
