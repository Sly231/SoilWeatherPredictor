# SoilWeatherPredictor
SoilWeatherPredictor is an AI-based tool that forecasts soil moisture using weather data like temperature, humidity, and precipitation. Powered by LSTM neural networks and SHAP for explainability, it helps optimize irrigation and crop management for better water resource use in agriculture.


### Step-by-Step Explanation of the Code:

1. **Loading and Preprocessing Data**:
   - Create synthetic datasets with columns like **soil moisture**, **temperature**, **humidity**, and **precipitation**, simulating data collected over time.
   - This merged data represents the input features (weather variables) and the target output (soil moisture) used to **train the model**.

2. **Scaling the Data**:
   - The features are standardized using **StandardScaler** so that each input (temperature, humidity, etc.) has a mean of 0 and a standard deviation of 1. This scaling is crucial for improving the performance of the **LSTM** model since it helps the network converge faster and learn effectively.

3. **Building the LSTM Model**:
   - The core of the model is an **LSTM (Long Short-Term Memory)** neural network, which is ideal for processing time-series data like weather patterns.
   - The LSTM layers capture **temporal dependencies** in the input data—i.e., how past weather affects current soil moisture levels.
   - **Dropout layers** are added to prevent overfitting by randomly setting input units to 0 during training.

4. **Reshaping Data**:
   - The data is reshaped into a 3D format that LSTM requires: (samples, timesteps, features). 
   - This step prepares the data for sequential learning, where each feature sequence (temperature, humidity, etc.) is passed through the LSTM one timestep at a time.

5. **Training the Model**:
   - The model is trained for **10 epochs** (cycles through the entire dataset) with a **batch size of 16**. 
   - It uses **mean squared error (MSE)** as the loss function, which measures the difference between predicted and actual soil moisture values.
   - After each epoch, the model's performance is evaluated on both the training set and validation set to ensure it generalizes well.

6. **Plotting Training Loss**:
   - A plot of training and validation loss helps visualize how well the model is learning.
   - **Training loss** should decrease over time, showing that the model is getting better at predicting soil moisture. **Validation loss** indicates how well the model performs on unseen data, helping avoid overfitting.

### SHAP (Explainability):

7. **Using SHAP (SHapley Additive exPlanations)**:
   - SHAP is used to explain **why** the model is making certain predictions by showing the contribution of each input feature (like temperature or humidity).
   - Reshape the data to a 2D format (samples, features) for SHAP analysis.
   - SHAP values provide insights into how much each feature affects the predicted soil moisture, offering **interpretability** to users who want to understand **how the model makes decisions**.

### Code Output:

- The **training and validation losses** printed in each epoch show how well the model is learning over time. In the training logs you shared:
  - **Loss** decreases as the model learns, though it may plateau or fluctuate.
  - **Validation loss** is important—it indicates whether the model is generalizing well to new data.
  
- **SHAP summary plot**:
  - Red and blue dots represent how **high and low values of input features** impact the model’s predictions.
  - Red dots mean high feature values (e.g., high temperature), while blue dots mean low feature values (e.g., low humidity). Their position along the x-axis shows their effect on soil moisture predictions—whether they push the prediction higher or lower.

This workflow enables our **SoilWeatherPredictor** to accurately forecast soil moisture based on weather patterns, providing useful insights for **irrigation planning** and **agriculture management**!
