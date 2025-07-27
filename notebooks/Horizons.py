'''
Apply uniform filters in time of different window lengths
'''
# %%
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
def predict_proba(train, test,predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6   ] = 0
    preds = pd.Series(preds, index=test.index, name ='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined
def backtest(df, model, predictors, start=2500, step = 250):
    all_predictions = []
    for i in range(start, len(df), step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:i+step].copy()
        predictions = predict_proba(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
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

# %%

horizons = [2,5,60,250,1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = df.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    df[ratio_column] = df['Close'] / rolling_averages['Close']

    trend_column = f"Trend_{horizon}"
    df[trend_column] = df.shift(1).rolling(horizon).sum()['Target']
    new_predictors += [ratio_column, trend_column]
df.dropna(inplace=True)
# %% 
predictors = ['Close', 'Open', 'High', 'Low', 'Volume']
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)
# %%
predictions = backtest(df, model, new_predictors)
# %%
predictions['Predictions'].value_counts()
# %%
precision_score(predictions['Target'], predictions['Predictions'])
# %%
