# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
processed_dir = os.path.join(basedir, 'data', 'processed')
processed_data_path = os.path.join(processed_dir, 'AAPL_cleaned.csv')
df = pd.read_csv(processed_data_path)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True) 



features = ['Open', 'High', 'Low', 'Close', 'Volume']


X = df[features]
y = df['Target']

# train-test split (use shuffle=False for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# %% test

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
# %%
