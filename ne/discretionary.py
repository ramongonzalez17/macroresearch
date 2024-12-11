import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fetch sector ETF data (for example, XLY for Consumer Discretionary, XLK for Technology, etc.)
def fetch_sector_data():
    sectors = ['XLY', 'XLK', 'XLU', 'XLC', 'XLI', 'XLE', 'XLB', 'XLF', 'XLP', 'XLI']
    sector_data = {}
    for sector in sectors:
        data = yf.download(sector, start="2000-01-01", end="2023-12-31")['Close']
        sector_data[sector] = data.resample('Q').last()  # Resample to quarterly frequency
    return sector_data

# Fetch macroeconomic data (Inflation and Unemployment) from the World Bank API
def fetch_macro_data():
    # Example using FRED or a similar API for inflation and unemployment
    inflation = pd.Series([0.02, 0.025, 0.03, 0.015], index=pd.date_range('2000-01-01', periods=4, freq='Q'))  # Sample inflation data
    unemployment = pd.Series([0.05, 0.045, 0.048, 0.04], index=pd.date_range('2000-01-01', periods=4, freq='Q'))  # Sample unemployment data
    
    # Resample and handle missing values or NaNs if needed
    inflation = inflation.resample('Q').ffill()  # Forward fill to handle missing quarterly data
    unemployment = unemployment.resample('Q').ffill()  # Forward fill to handle missing quarterly data
    
    return inflation, unemployment

# Function to check for missing values or infinities and handle them
def clean_data(df):
    # Check for NaNs or Inf and drop rows with any missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
    df = df.dropna()  # Drop rows with NaN values
    return df

# Preprocess Data and Run OLS Regression
def run_regression(sector_data, inflation, unemployment):
    """Run OLS regression on sector returns against inflation and unemployment."""
    results = {}
    for sector, data in sector_data.items():
        sector_returns = data.pct_change().dropna()  # Quarterly returns
        
        # Merge data into a single DataFrame
        merged_data = pd.concat([sector_returns, inflation, unemployment], axis=1)
        merged_data.columns = ['Sector Returns', 'Inflation', 'Unemployment']
        
        # Clean the data (remove NaNs and infinite values)
        merged_data = clean_data(merged_data)
        
        # Debugging: Print out the shape of the data to check if any rows are dropped
        print(f"Sector: {sector}")
        print(f"Shape of merged_data: {merged_data.shape}")
        
        # If after cleaning, the data is empty, we can't run the regression
        if merged_data.empty:
            print(f"Warning: No valid data available for sector: {sector}")
            continue
        
        # OLS Regression
        X = merged_data[['Inflation', 'Unemployment']]
        X = sm.add_constant(X)  # Add constant term for the intercept
        y = merged_data['Sector Returns']
        
        # Debugging: Ensure no empty arrays are passed
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        
        model = sm.OLS(y, X).fit()  # Fit the model
        results[sector] = model
        
        # Plotting Actual vs Predicted Returns for the sector
        plt.figure(figsize=(10, 6))
        plt.plot(y.index, y, label="Actual Returns", marker='o')
        plt.plot(y.index, model.fittedvalues, label="Predicted Returns", linestyle='--')
        plt.title(f"Actual vs Predicted Returns for {sector}")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()
        
        # Visualize the coefficients for the regression
        coef = model.params
        plt.figure(figsize=(8, 4))
        coef.plot(kind='bar', title=f"Regression Coefficients for {sector}")
        plt.xlabel("Variables")
        plt.ylabel("Coefficient Value")
        plt.show()
        
    return results

# Fetch Data
sector_data = fetch_sector_data()
inflation, unemployment = fetch_macro_data()

# Run the regression
regression_results = run_regression(sector_data, inflation, unemployment)

# Print regression results
for sector, result in regression_results.items():
    print(f"Sector: {sector}")
    print(result.summary())
