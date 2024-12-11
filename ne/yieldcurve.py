import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fetch Utilities ETF (XLU) Data
def fetch_xlu_data():
    """Fetch historical XLU data and calculate quarterly returns."""
    data = yf.download('XLU', start='2000-01-01', end='2023-12-31')['Close']
    data = data.resample('Q').last().pct_change()  # Quarterly returns
    return data

# Fetch Yield Curve Data (10Y-2Y Spread)
def fetch_yield_curve():
    """Fetch 10Y-2Y Treasury yield spread data."""
    # Fetch 10-Year (^TNX) and 2-Year (^IRX) yields from Yahoo Finance
    yield_data = yf.download(['^TNX', '^IRX'], start='2000-01-01', end='2023-12-31')['Close']
    yield_data['10Y-2Y Spread'] = yield_data['^TNX'] - yield_data['^IRX']  # Calculate spread
    yield_data = yield_data['10Y-2Y Spread'].resample('Q').last()  # Resample to quarterly
    return yield_data

# Fetch Unemployment Rate (Simulated for Simplicity)
def fetch_unemployment():
    """Fetch dummy unemployment data."""
    dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='Q')
    unemployment = pd.Series(data=(5 - 0.1 * (dates.year - 2000)), index=dates)  # Dummy Unemployment
    return unemployment

# Prepare Data
def prepare_data_with_fewer_features(xlu_data, yield_curve, unemployment):
    """Prepare data with only selected features."""
    # Combine relevant features
    data = pd.concat([xlu_data, yield_curve, unemployment], axis=1)
    data.columns = ['XLU_Returns', '10Y-2Y Spread', 'Unemployment']
    data.dropna(inplace=True)  # Drop rows with missing values

    # Features and target
    X = data.drop(columns=['XLU_Returns'])
    y = data['XLU_Returns']
    return X, y

# Fetch data
xlu_data = fetch_xlu_data()
yield_curve = fetch_yield_curve()
unemployment = fetch_unemployment()

# Prepare data
X, y = prepare_data_with_fewer_features(xlu_data, yield_curve, unemployment)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualization of Actual vs Predicted Returns
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual XLU Returns', marker='o')
plt.plot(y_pred, label='Predicted XLU Returns', marker='x')
plt.legend()
plt.title('Actual vs Predicted XLU Returns (Utilities Sector)')
plt.xlabel('Time')
plt.ylabel('XLU Quarterly Returns')
plt.grid(True)
plt.show()

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(X.columns, importances, color='blue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()
