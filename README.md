**SoilWeatherPredictor**
SoilWeatherPredictor is an AI-powered tool designed to forecast soil moisture levels using real-time weather data, while also recommending the most suitable crops for specific locations across the United States. Leveraging Long Short-Term Memory (LSTM) neural networks for soil moisture prediction and SHAP (SHapley Additive exPlanations) for explainability, this model helps optimize agricultural practices. It takes into account temperature, precipitation, and soil moisture to recommend crops that are best suited for current weather conditions, aiding in irrigation and crop planning for farmers.

**Features**
**Soil Moisture Prediction:**
Uses real-time weather data from Meteomatics API to predict soil moisture levels based on key weather inputs such as temperature, precipitation, and wind speed. The LSTM-based model is trained to recognize patterns in historical weather data for accurate moisture forecasts.

**Crop Recommendations:**
Based on current weather conditions (temperature, precipitation, and moisture levels), the model provides a stochastic recommendation of suitable crops for different US cities. The crop recommendations are drawn from a variety of crops like corn, wheat, soybeans, barley, and more, with some randomness introduced to account for varying conditions.

**Weather Forecast Visualization:**
A Plotly-based interactive plot is generated for each city, displaying the forecasted weather parameters (temperature, precipitation, and wind speed) over the next seven days. This helps visualize trends in weather conditions for better planning.

**Explainable AI:**
The model utilizes SHAP to explain how different weather features contribute to soil moisture predictions. A SHAP summary plot is generated for each city, providing insights into the impact of different weather parameters on the model’s predictions.

**How It Works**
**Weather Data Collection:**
The model fetches real-time weather data from the Meteomatics API for major US cities, including New York, Los Angeles, Chicago, and others. This data includes hourly temperature, precipitation, and wind speed forecasts for the next seven days.

**Soil Moisture Prediction:**
After gathering the weather data, the model uses an LSTM neural network to predict soil moisture levels for each location. The model is trained on past weather and synthetic soil moisture data and can capture temporal patterns effectively.

**Crop Recommendation:**
Based on the predicted soil moisture, along with real-time temperature and precipitation data, the model recommends suitable crops. These recommendations are made using pre-defined crop suitability thresholds and introduce slight randomness to account for variable conditions, ensuring stochasticity in the recommendations.

**Explainability:**
SHAP is used to explain the model's predictions, making the results more interpretable. This ensures that users understand the reasoning behind each soil moisture prediction.

**Weather Visualization:**
A Plotly-based interactive plot visualizes the weather forecast for each city, including temperature, precipitation, and wind speed over time.

**Key Technologies**
**Keras (LSTM)**
Long Short-Term Memory (LSTM) neural networks are used to model the time series data of weather conditions and predict soil moisture levels.

**Meteomatics API**
Provides real-time weather data for various US cities, ensuring the model uses accurate, up-to-date information.

**SHAP (SHapley Additive exPlanations)**
Offers explainability for the model’s predictions by providing insights into how different features (temperature, precipitation, and wind speed) affect the outcome.

**Plotly**
For creating interactive plots that visualize weather conditions in an easily understandable format.
