import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

fed_funds = pd.read_csv(
    'C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/fed_funds_target_rate(1990-2024).csv',
    names=['Date', 'Rate'],
    skiprows=1
)
russell_2000 = pd.read_csv(
    'C:/Users/gonza/OneDrive/Desktop/Macro Research/macroresearch/data/russell_2000(1992-2024).csv',
    names=['Date', 'Index'],
    skiprows=1
)

fed_funds['Date'] = pd.to_datetime(fed_funds['Date'], format='%b-%d-%Y')
russell_2000['Date'] = pd.to_datetime(russell_2000['Date'], format='%b-%d-%Y')


fed_funds.set_index('Date', inplace=True)
russell_2000.set_index('Date', inplace=True)

fed_funds_monthly = fed_funds.resample('MS').mean()
russell_monthly = russell_2000.resample('MS').mean()

ccf_vals = ccf(russell_monthly['Index'], fed_funds_monthly['Rate'], adjusted=False)

plt.figure(figsize=(12, 6))
plt.stem(range(len(ccf_vals)), ccf_vals)
plt.title('Cross-Correlation between Federal Funds Rate and Russell 2000 Index')
plt.xlabel('Lags (Months)')
plt.ylabel('Cross-Correlation Coefficient')
plt.show()
