import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

fed_funds = pd.read_csv('data/fed_funds_target_rate(1990-2024).csv', names=['Date', 'Rate'], skiprows=1)
russell_2000 = pd.read_csv('data/russell_2000(1992-2024).csv', names=['Date', 'Index'], skiprows=1)

fed_funds['Date'] = pd.to_datetime(fed_funds['Date'], format='%b-%d-%Y')
russell_2000['Date'] = pd.to_datetime(russell_2000['Date'], format='%b-%d-%Y')
fed_funds.set_index('Date', inplace=True)
russell_2000.set_index('Date', inplace=True)

fed_funds_q = fed_funds.resample('Q').mean()
russell_2000_q = russell_2000.resample('Q').mean()

fed_funds_q['Fed Funds Change'] = fed_funds_q['Rate'].pct_change() * 100
russell_2000_q['Russell Change'] = russell_2000_q['Index'].pct_change() * 100

combined_data = pd.concat([fed_funds_q['Fed Funds Change'], russell_2000_q['Russell Change']], axis=1).dropna()

X = combined_data['Fed Funds Change']  
y = combined_data['Russell Change']   
X = sm.add_constant(X)  

model = sm.OLS(y, X).fit()

print(model.summary())

plt.figure(figsize=(14, 7))
plt.plot(fed_funds_q.index, fed_funds_q['Fed Funds Change'], label='Federal Funds Rate (% Change)', color='blue')
plt.plot(russell_2000_q.index, russell_2000_q['Russell Change'], label='Russell 2000 (% Change)', color='green')
plt.title('Quarter-over-Quarter Percentage Changes')
plt.xlabel('Year')
plt.ylabel('Percentage Change')
plt.axhline(0, color='black', linewidth=0.5) 
plt.legend()
plt.grid(True)
plt.show()
