'''
Here we build a backtesting system to automatically
train a model on N years to predict the next year.
'''
# %%
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
def predict(train, test,predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name ='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined
def backtest(df, model, predictors, start=2500, step = 250):
    all_predictions = []
    for i in range(start, len(df), step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:i+step].copy()
        predictions = predict(train, test, predictors, model)
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
predictors = ['Close', 'Open', 'High', 'Low', 'Volume']
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=42)
# %%

predictions = backtest(df, model, predictors)
predictions['Predictions'].value_counts()
precision_score(predictions['Target'], predictions['Predictions'])
# %%
