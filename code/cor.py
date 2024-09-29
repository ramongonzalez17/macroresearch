import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

fed_funds = pd.read_csv('C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/fed_funds_target_rate(1990-2024).csv', names=['Date', 'Value'], skiprows=1)
russell_2000 = pd.read_csv('C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/russell_2000(1992-2024).csv', names=['Date', 'Value'], skiprows=1)

fed_funds['Date'] = pd.to_datetime(fed_funds['Date'], format='%b-%d-%Y')
russell_2000['Date'] = pd.to_datetime(russell_2000['Date'], format='%b-%d-%Y')
fed_funds.set_index('Date', inplace=True)
russell_2000.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(fed_funds['Value'], ax=plt.gca(), title='Autocorrelation of Federal Funds Target Rate')
plt.grid(True)

plt.subplot(212)
plot_acf(russell_2000['Value'], ax=plt.gca(), title='Autocorrelation of Russell 2000 Index')
plt.grid(True)

plt.tight_layout()
plt.show()
