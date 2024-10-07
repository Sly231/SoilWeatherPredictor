# SoilWeatherPredictor

**SoilWeatherPredictor** is an AI-powered tool designed to forecast soil moisture levels using real-time weather data, while also recommending the most suitable crops for specific locations across the United States. Leveraging Long Short-Term Memory (LSTM) neural networks for soil moisture prediction, this model helps optimize agricultural practices. It takes into account temperature, precipitation, and soil moisture to recommend crops that are best suited for current weather conditions, aiding in irrigation and crop planning for farmers.

## Features

- **Soil Moisture Prediction**: Uses real-time weather data from the **Meteomatics API** to predict soil moisture levels based on key weather inputs such as temperature, precipitation, and wind speed. The LSTM-based model is trained to recognize patterns in historical weather data for accurate moisture forecasts.

- **Crop Recommendations**: Based on current weather conditions (temperature, precipitation, and moisture levels), the model provides deterministic recommendations for suitable crops in different US cities. The recommendations are drawn from a variety of crops like corn, wheat, soybeans, barley, and more, based on strict suitability thresholds for temperature, precipitation, and soil moisture.

- **Weather Forecast Visualization**: A Plotly-based interactive plot is generated for each city, displaying the forecasted weather parameters (temperature, precipitation, and wind speed) over the next seven days. This helps visualize trends in weather conditions for better planning.

- **Detailed Logging**: During the crop recommendation process, detailed logs are printed to the console showing the environmental conditions being checked (temperature, precipitation, soil moisture) and how they align with crop thresholds. This ensures transparency in the recommendation process.

## How It Works

1. **Weather Data Collection**: The model fetches real-time weather data from the Meteomatics API for major US cities, including New York, Los Angeles, Chicago, and others. This data includes hourly temperature, precipitation, and wind speed forecasts for the next seven days.

2. **Soil Moisture Prediction**: After gathering the weather data, the model uses an LSTM neural network to predict soil moisture levels for each location. The model is trained on past weather and synthetic soil moisture data and can capture temporal patterns effectively.

3. **Crop Recommendation**: Based on the predicted soil moisture, along with real-time temperature and precipitation data, the model recommends suitable crops.  These recommendations are made using pre-defined crop suitability thresholds, checking for temperature, precipitation, and moisture requirements for each crop.

4. **Weather Visualization**: A Plotly-based interactive plot visualizes the weather forecast for each city, including temperature, precipitation, and wind speed over time.

## Key Technologies

- **Keras (LSTM)**: Long Short-Term Memory (LSTM) neural networks are used to model the time series data of weather conditions and predict soil moisture levels.

- **Meteomatics API**: Provides real-time weather data for various US cities, ensuring the model uses accurate, up-to-date information.

- **Plotly**: Used for creating interactive plots that visualize weather conditions in an easily understandable format.
