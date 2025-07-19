# %%
import pandas as pd
import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_data_path = os.path.join(basedir, 'data', 'raw', 'AAPL.csv')
processed_dir = os.path.join(basedir, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)
processed_data_path = os.path.join(processed_dir, 'AAPL_cleaned.csv')

# Load raw data
df = pd.read_csv(raw_data_path)
# %%
# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sort by date
df = df.sort_index()

df = df.dropna()
# %%
df.to_csv(processed_data_path)

print(f"✅ Cleaned data saved to: {processed_data_path}")
print(df.head())
# %%
