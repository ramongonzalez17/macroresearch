import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

russell_data = pd.read_csv(r'C:\Users\gonza\OneDrive\Desktop\Macro Research\macroresearch\data\russell_2000(1992-2024).csv', names=['Date', 'Value'], header=None)
fed_funds_data = pd.read_csv(r'C:\Users\gonza\OneDrive\Desktop\Macro Research\macroresearch\data\fed_funds_target_rate(1990-2024).csv', names=['Date', 'Value'], header=None)

russell_data['Value'] = pd.to_numeric(russell_data['Value'], errors='coerce')
fed_funds_data['Value'] = pd.to_numeric(fed_funds_data['Value'], errors='coerce')


russell_data['Date'] = pd.to_datetime(russell_data['Date'], format='%b-%d-%Y', errors='coerce')
fed_funds_data['Date'] = pd.to_datetime(fed_funds_data['Date'], format='%b-%d-%Y', errors='coerce')


russell_data.dropna(subset=['Date', 'Value'], inplace=True)
fed_funds_data.dropna(subset=['Date', 'Value'], inplace=True)

russell_data.sort_values('Date', inplace=True)
fed_funds_data.sort_values('Date', inplace=True)
russell_data.set_index('Date', inplace=True)
fed_funds_data.set_index('Date', inplace=True)


russell_data['Value'].plot(title='Russell 2000 Index Over Time')
plt.show()
fed_funds_data['Value'].plot(title='Federal Funds Target Rate Over Time')
plt.show()


plot_acf(russell_data['Value'].diff().dropna())
plot_pacf(russell_data['Value'].diff().dropna())
plt.show()

model = ARIMA(russell_data['Value'], order=(1,1,1))  
results = model.fit()

print(results.summary())
residuals = results.resid
residuals.plot(title='Residuals')
plt.show()
residuals.plot(kind='kde', title='Density of Residuals')
plt.show()

# Forecasting
forecast = results.get_forecast(steps=12)
forecast_summary = forecast.summary_frame()
print(forecast_summary)
