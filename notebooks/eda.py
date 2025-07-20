# %% Exploratory data analysis (EDA) for the project

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
processed_dir = os.path.join(basedir, 'data', 'processed')
processed_data_path = os.path.join(processed_dir, 'AAPL_cleaned.csv')
df = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)


# %% Closing price
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], marker='o', linestyle='-', label='Close')

plt.title('Apple Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %% Volume of trades
plt.figure(figsize=(14,4))
plt.plot(df.index, df['Volume'], marker='o', linestyle='-', label='Close')
plt.title('Apple Stock Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %% Daily returns histogram
df['Daily Return'] = df['Close'].pct_change()  # percentage change from previous day

plt.figure(figsize=(14,5))
df['Daily Return'].hist(bins=100, alpha=0.7)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(14,6))
plt.plot(df['Daily Return'])
plt.title('Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.show()
# %%Correlation and covariance matrices
price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']

corr_matrix = df[price_cols].corr()
cov_matrix = df[price_cols].cov()

print("Correlation matrix:\n", corr_matrix)
print("\nCovariance matrix:\n", cov_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Price Features')
plt.show()

# %%
