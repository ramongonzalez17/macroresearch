import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.formula.api import ols

fed_funds = pd.read_csv('C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/fed_funds_target_rate(1990-2024).csv', names=['Date', 'FedFunds'], skiprows=1)
russell_2000 = pd.read_csv('C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/russell_2000(1992-2024).csv', names=['Date', 'Russell2000'], skiprows=1)
gdp = pd.read_csv('C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/gdp.csv', names=['Date', 'GDPGrowth'], skiprows=1)

fed_funds['Date'] = pd.to_datetime(fed_funds['Date'], errors='coerce')
russell_2000['Date'] = pd.to_datetime(russell_2000['Date'], errors='coerce')
gdp['Date'] = pd.to_datetime(gdp['Date'], errors='coerce')

fed_funds.set_index('Date', inplace=True)
russell_2000.set_index('Date', inplace=True)
gdp.set_index('Date', inplace=True)

start_date = pd.Timestamp('1992-01-01')
fed_funds = fed_funds[fed_funds.index >= start_date]
russell_2000 = russell_2000[russell_2000.index >= start_date]
gdp = gdp[gdp.index >= start_date]

fed_funds_q = fed_funds.resample('Q').mean()
russell_2000_q = russell_2000.resample('Q').mean()
gdp_q = gdp.resample('Q').mean()

data = pd.concat([fed_funds_q, russell_2000_q, gdp_q], axis=1).dropna()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
variables = ['FedFunds', 'Russell2000', 'GDPGrowth']
for i, var in enumerate(variables):
    plot_acf(data[var], ax=axes[i, 0], title=f'ACF of {var}')
    plot_pacf(data[var], ax=axes[i, 1], title=f'PACF of {var}')
plt.tight_layout()
plt.show()

# Stationarity test
for var in variables:
    result = adfuller(data[var])
    print(f'ADF Statistic for {var}: {result[0]}')
    print(f'p-value: {result[1]}')

# Linear regression
model = ols('Russell2000 ~ FedFunds + GDPGrowth', data=data).fit()
print(model.summary())

# Plot residuals
plt.figure(figsize=(8, 4))
plt.plot(model.resid)
plt.title('Residuals from Regression Model')
plt.show()

# Check residual normality
plt.figure(figsize=(8, 4))
model.resid.plot(kind='kde')
plt.title('Density of Residuals')
plt.show()
