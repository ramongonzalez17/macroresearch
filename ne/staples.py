import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import wbgapi as wb  # World Bank API


# Fetch XLP (Consumer Staples ETF) Data
def fetch_xlp_data():
    """Fetch historical XLP data (Consumer Staples ETF)."""
    data = yf.download('XLP', start='2000-01-01', end='2023-12-31')['Close']
    data = data.resample('Q').last()  # Resample to quarterly frequency
    return data


# Fetch GDP Growth Data
def fetch_gdp_growth():
    """Fetch GDP growth data from the World Bank API."""
    gdp_data = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG', 'USA', time=range(2000, 2024)).T
    gdp_data.index = gdp_data.index.str.replace('YR', '').astype(int)
    gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y')
    gdp_data = gdp_data.resample('Q').ffill()  # Resample to quarterly frequency
    return gdp_data.squeeze()  # Return as a pandas Series


# Fetch Inflation and Unemployment Data
def fetch_inflation_and_unemployment():
    """Fetch inflation and unemployment data."""
    # Fetch inflation rate (% change in CPI)
    inflation = wb.data.DataFrame('FP.CPI.TOTL.ZG', 'USA', time=range(2000, 2024)).T
    inflation.index = inflation.index.str.replace('YR', '').astype(int)
    inflation.index = pd.to_datetime(inflation.index, format='%Y')
    inflation = inflation.resample('Q').ffill()  # Resample to quarterly frequency
    
    # Fetch unemployment rate (% of total labor force)
    unemployment = wb.data.DataFrame('SL.UEM.TOTL.ZS', 'USA', time=range(2000, 2024)).T
    unemployment.index = unemployment.index.str.replace('YR', '').astype(int)
    unemployment.index = pd.to_datetime(unemployment.index, format='%Y')
    unemployment = unemployment.resample('Q').ffill()  # Resample to quarterly frequency
    
    return inflation.squeeze(), unemployment.squeeze()


# Prepare Dataset with Features
def prepare_xlp_data_with_features(xlp_data, gdp_data, inflation, unemployment):
    """Prepare dataset with additional macroeconomic features."""
    # Combine XLP, GDP growth, and additional features
    data = pd.concat([
        xlp_data.pct_change().dropna(),  # Percentage change of XLP (target)
        gdp_data,
        inflation,
        unemployment
    ], axis=1)
    data.columns = ['XLP_Returns', 'GDP_Growth', 'Inflation_Rate', 'Unemployment_Rate']
    data.dropna(inplace=True)  # Drop rows with missing values
    return data


# Add Lagged Features
def add_lagged_features(data, lags=4):
    """Add lagged features to the dataset."""
    for lag in range(1, lags + 1):
        data[f'GDP_Growth_Lag{lag}'] = data['GDP_Growth'].shift(lag)
        data[f'Inflation_Rate_Lag{lag}'] = data['Inflation_Rate'].shift(lag)
        data[f'Unemployment_Rate_Lag{lag}'] = data['Unemployment_Rate'].shift(lag)
    data.dropna(inplace=True)  # Drop rows with NaN values after adding lags
    return data


# Fetch Data
xlp_data = fetch_xlp_data()
gdp_data = fetch_gdp_growth()
inflation, unemployment = fetch_inflation_and_unemployment()

# Prepare Data
data = prepare_xlp_data_with_features(xlp_data, gdp_data, inflation, unemployment)
data = add_lagged_features(data)

# Define Features and Target
X = data.drop(['XLP_Returns'], axis=1)  # Exclude target
y = data['XLP_Returns']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train Gradient Boosting Regressor
gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)

# Make Predictions
y_pred = gbm.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature Importance
importance = gbm.feature_importances_
for i, feature in enumerate(X.columns):
    print(f"Feature: {feature}, Importance: {importance[i]:.4f}")

# Visualize Results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual XLP Returns")
plt.plot(y_pred, label="Predicted XLP Returns")
plt.title("Actual vs Predicted XLP Returns (Consumer Staples)")
plt.xlabel("Time")
plt.ylabel("XLP Returns")
plt.legend()
plt.show()
