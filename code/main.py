import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/fed_funds_target_rate(1990-2024).csv', names=['Date', 'Federal Funds Rate'], skiprows=1)

df['Date'] = pd.to_datetime(df['Date'], format='%b-%d-%Y') 

df.set_index('Date', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Federal Funds Rate'], color='blue', label='Federal Funds Rate')
plt.title('Federal Funds Rate (Daily) from 1990 to 2024')
plt.xlabel('Year')
plt.ylabel('Federal Funds Rate (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
