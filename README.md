# Finance Stock Price Prediction

## Overview
This (work in progres!) project demonstrates a complete data science workflow applied to financial stock data. The focus is on showcasing skills in data collection, cleaning, exploratory data analysis, feature engineering, dimensionality reduction with PCA, and predictive modeling using Python.

The dataset consists of historical daily stock prices for Apple Inc. (AAPL), downloaded from Yahoo Finance. The goal is to build a model that predicts stock returns using principal components derived from financial features.

## Project Structure
finance-stock-prediction/
│
├── data/
│ ├── raw/ # Raw downloaded stock price CSV files
│ └── processed/ # Cleaned and feature-engineered datasets
│
├── notebooks/ # Jupyter notebooks for exploration and analysis
│
├── results/
│ ├── figures/ # Plots and visualizations
│ └── models/ # Trained model files
│
├── src/ # Source code modules
│ ├── data_collection.py # Script to download stock data with yfinance
│ ├── data_cleaning.py # Data cleaning and preprocessing functions
│ ├── features.py # Feature engineering and PCA analysis
│ ├── train.py # Model training scripts
│ └── evaluate.py # Model evaluation and testing
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── main.py # Optional entry point script


## Features
- Data download and storage of raw financial data using `yfinance`.
- Handling missing data and preprocessing time series.
- Exploratory data analysis with visualizations (price trends, returns, covariance).
- Feature engineering: moving averages, rolling volatility, lagged returns.
- Principal Component Analysis (PCA) for dimensionality reduction.
- Linear Regression model to predict future stock returns.
- Evaluation of model performance on training and test sets.
- Modular Python code and well-documented Jupyter notebooks for reproducibility.

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- yfinance

Install dependencies with:

```bash
pip install -r requirements.txt

Clone this repository:
git clone https://github.com/smaniot96/finance-stock-prediction.git
cd finance-stock-prediction


Download stock data:
python src/data_collection.py

Follow the notebooks or run scripts to clean data, engineer features, train models, and evaluate results.