# %%
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# %% Import data
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
processed_dir = os.path.join(basedir, 'data', 'processed')
processed_data_path = os.path.join(processed_dir, 'AAPL_cleaned.csv')
df = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)


# %% create tomorrow's close price
df['Tomorrow'] = df['Close'].shift(-1)
df.dropna(inplace=True)
# Create bolean target variable
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

# %% train random forest classifier
model = RandomForestClassifier(n_estimators = 100, min_samples_split=100, random_state = 42)
train = df.iloc[:-100]  # all but last 100 rows
test = df.iloc[-100:]  # last 100 rows
predictors = ['Close', 'Open', 'High', 'Low', 'Volume']
model.fit(train[predictors], train['Target'])
# %% Evaluate model
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test['Target'], preds)
# %%
