import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import wbgapi as wb  # World Bank API

# Fetch Energy Sector ETF (XLE) Data
def fetch_energy_etf():
    """Fetch historical XLE data (Energy ETF)."""
    data = yf.download('XLE', start='2000-01-01', end='2023-12-31')['Close']
    data = data.resample('Q').last()  # Resample to quarterly frequency
    return data

# Fetch Crude Oil Prices
def fetch_crude_oil_prices():
    """Fetch crude oil prices (WTI) from Yahoo Finance."""
    data = yf.download('CL=F', start='2000-01-01', end='2023-12-31')['Close']
    data = data.resample('Q').last()  # Resample to quarterly frequency
    return data

# Fetch GDP Growth Data
def fetch_gdp_growth():
    """Fetch GDP growth data from the World Bank API."""
    gdp_data = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG', 'USA', time=range(2000, 2024))
    gdp_data = gdp_data.T  # Transpose to make years rows
    gdp_data.index = gdp_data.index.str.replace('YR', '')  # Remove 'YR' prefix
    gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y')  # Convert to datetime
    gdp_data = gdp_data.resample('Q').ffill()  # Resample to quarterly frequency
    return gdp_data.squeeze()  # Return as a pandas Series

# Prepare Data
def prepare_data(xle_data, oil_prices, gdp_data):
    """Merge and preprocess the data."""
    # Merge data into a single DataFrame
    combined = pd.concat([xle_data, oil_prices, gdp_data], axis=1)
    combined.columns = ['XLE', 'Crude_Oil', 'GDP']
    combined.dropna(inplace=True)  # Drop rows with missing values

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined)

    # Create input features (X) and target (y)
    X, y = [], []
    lookback = 4  # Use the last 4 quarters as input
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])  # Last 4 quarters as features
        y.append(scaled_data[i, 0])  # XLE (Energy ETF) as target
    X, y = np.array(X), np.array(y)
    return X, y, scaler, combined

# Fetch data
xle_data = fetch_energy_etf()
oil_prices = fetch_crude_oil_prices()
gdp_data = fetch_gdp_growth()

# Prepare data
X, y, scaler, combined_data = prepare_data(xle_data, oil_prices, gdp_data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define the LSTM Model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Single output for XLE performance
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the Model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make Predictions
y_pred = model.predict(X_test)

# Unscale predictions and actual values
y_test_unscaled = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 2))), axis=1)
)[:, 0]
y_pred_unscaled = scaler.inverse_transform(
    np.concatenate((y_pred, np.zeros((len(y_pred), 2))), axis=1)
)[:, 0]

# Visualize Results
plt.figure(figsize=(10, 6))
plt.plot(y_test_unscaled, label="Actual XLE Performance")
plt.plot(y_pred_unscaled, label="Predicted XLE Performance")
plt.legend()
plt.title("Actual vs Predicted XLE Performance (Energy Sector)")
plt.xlabel("Time")
plt.ylabel("XLE Performance")
plt.show()
