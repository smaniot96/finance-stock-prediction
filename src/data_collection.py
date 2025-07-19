# %%
import yfinance as yf
import os
import pandas as pd

# Get project base directory (one level up from this script)
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-12-31'

# Construct full path to data/raw inside project root
raw_data_dir = os.path.join(basedir, 'data', 'raw')
os.makedirs(raw_data_dir, exist_ok=True)

print(f"Downloading {ticker} data...")
data = yf.download(ticker, start=start_date, end=end_date)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
print(data.columns)  

# %%

csv_path = os.path.join(raw_data_dir, f"{ticker}.csv")
data.to_csv(csv_path)
print(f"Saved {ticker} data to {csv_path} ({len(data)} rows)")
