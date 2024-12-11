import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fetch XLK Data (Technology Sector)
def fetch_xlk_data():
    """Fetch historical XLK data and calculate quarterly returns."""
    xlk_data = yf.download('XLK', start='2000-01-01', end='2023-12-31')['Close']
    xlk_returns = xlk_data.resample('Q').last().pct_change()  # Quarterly returns
    return xlk_returns

# Fetch Treasury Yields (10Y and 2Y)
def fetch_treasury_yields():
    """Fetch Treasury yields and calculate quarterly changes."""
    treasury_data = yf.download(['^TNX', '^IRX'], start='2000-01-01', end='2023-12-31')['Close']
    treasury_data['10Y-2Y Spread'] = treasury_data['^TNX'] - treasury_data['^IRX']  # Yield spread
    treasury_data = treasury_data.resample('Q').last().pct_change()  # Quarterly changes
    return treasury_data[['^TNX', '^IRX', '10Y-2Y Spread']].rename(columns={'^TNX': '10Y', '^IRX': '2Y'})

# Prepare Data for Analysis
def prepare_data(xlk_returns, yield_data):
    """Merge XLK returns and Treasury yield data."""
    combined_data = pd.concat([xlk_returns, yield_data], axis=1)
    combined_data.columns = ['XLK Returns', '10Y', '2Y', '10Y-2Y Spread']
    combined_data.dropna(inplace=True)
    return combined_data

# Run Regression Analysis
def run_regression(data):
    """Perform regression analysis to assess sensitivity of XLK to interest rates."""
    X = data[['10Y', '2Y', '10Y-2Y Spread']]
    y = data['XLK Returns']
    X = sm.add_constant(X)  # Add constant for regression
    model = sm.OLS(y, X).fit()
    return model

# Fetch and prepare data
xlk_returns = fetch_xlk_data()
yield_data = fetch_treasury_yields()
data = prepare_data(xlk_returns, yield_data)

# Perform regression
model = run_regression(data)

# Print regression summary
print(model.summary())

# Visualize Results: Actual vs Predicted XLK Returns
data['Predicted Returns'] = model.predict(sm.add_constant(data[['10Y', '2Y', '10Y-2Y Spread']]))
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['XLK Returns'], label='Actual XLK Returns', marker='o')
plt.plot(data.index, data['Predicted Returns'], label='Predicted XLK Returns', marker='x')
plt.title('Actual vs Predicted XLK Returns (Technology Sector)')
plt.xlabel('Time')
plt.ylabel('XLK Quarterly Returns')
plt.legend()
plt.grid()
plt.show()
