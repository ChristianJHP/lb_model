# Improved Lumber Futures Trading Model with GRU Neural Network

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from pytrends.request import TrendReq
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Fixing deprecated warning with pct_change()
pd.options.mode.use_inf_as_na = True

# Define tickers and data sources
TICKERS = {
    'WOOD': 'iShares Global Timber & Forestry ETF',
    'UUP': 'Invesco DB US Dollar Index Bullish Fund',
    '^TNX': '10-Year Treasury Yield'
}

# Fetch data with fixed warnings
def fetch_data():
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2025, 5, 19)
    df = pd.DataFrame()
    for ticker in TICKERS.keys():
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in data.columns:
            df[ticker] = data['Adj Close'].ffill()
        elif 'Close' in data.columns:
            df[ticker] = data['Close'].ffill()
        else:
            print(f"Warning: No price data found for {ticker}")
    housing_starts = pdr.get_data_fred('HOUST', start=start_date, end=end_date)
    df['Housing_Starts'] = housing_starts['HOUST'].resample('D').ffill()
    # Fetch Google Trends data and merge
    trends = get_or_fetch_trends('lumber price', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = df.merge(trends, left_index=True, right_index=True, how='left')
    df['GoogleTrends_Lumber'] = df['GoogleTrends_Lumber'].ffill()
    df = df.dropna()
    return df

# Enhanced feature engineering

def add_features(df):
    features = pd.DataFrame(index=df.index)
    features['Return_5d'] = df['WOOD'].pct_change(5)
    features['RSI_14'] = RSIIndicator(df['WOOD'], window=14).rsi()
    features['ATR_14'] = AverageTrueRange(
        high=df['WOOD'], low=df['WOOD'], close=df['WOOD'], window=14
    ).average_true_range()
    features['GoogleTrends_Lumber'] = df['GoogleTrends_Lumber']
    # features['NDVI'] = df['NDVI']  # Temporarily ignore NDVI until dataset is available
    # Housing Starts features
    features['Housing_Starts'] = df['Housing_Starts']
    features['Housing_Starts_Pct_Change'] = df['Housing_Starts'].pct_change()
    features['Housing_Starts_MA7'] = df['Housing_Starts'].rolling(window=7).mean()
    features['Housing_Starts_MA30'] = df['Housing_Starts'].rolling(window=30).mean()
    # DXY (US Dollar Index) features using UUP as proxy
    features['DXY'] = df['UUP']
    features['DXY_Pct_Change'] = df['UUP'].pct_change()
    # CME Open Interest and Volume (assume columns exist in df)
    if 'CME_Volume' in df.columns:
        features['CME_Volume'] = df['CME_Volume']
        features['CME_Volume_Pct_Change'] = df['CME_Volume'].pct_change()
    if 'CME_OpenInterest' in df.columns:
        features['CME_OpenInterest'] = df['CME_OpenInterest']
        features['CME_OpenInterest_Pct_Change'] = df['CME_OpenInterest'].pct_change()
    # Seasonal indicators
    features['Month'] = features.index.month
    features['Quarter'] = features.index.quarter
    features['Target'] = np.where(df['WOOD'].pct_change(5).shift(-5) > 0.01, 1, 0)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features

# GRU model for time series prediction
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation
def train_gru(X_train, y_train, X_test, y_test):
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    return model, history

# Main function
def main():
    df = fetch_data()
    features = add_features(df)
    X = features.drop(columns=['Target']).values
    y = features['Target'].values
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    time_series_cv(X, y)
    # plot_trends_vs_price(df)

def get_or_fetch_trends(keyword, start_date, end_date, filename='lumber_trends.csv'):
    if os.path.exists(filename):
        trends = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        trends = fetch_google_trends(keyword, start_date, end_date)
        trends.to_csv(filename)
    return trends

def fetch_google_trends(keyword, start_date, end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [keyword]
    pytrends.build_payload(kw_list, cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    trends = pytrends.interest_over_time()
    if 'isPartial' in trends.columns:
        trends = trends.drop(columns=['isPartial'])
    trends = trends.rename(columns={keyword: 'GoogleTrends_Lumber'})
    return trends

def plot_trends_vs_price(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['WOOD'], color='tab:blue', label='WOOD Price')
    ax1.set_ylabel('WOOD Price', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['GoogleTrends_Lumber'], color='tab:orange', label='Google Trends (Lumber Price)', alpha=0.6)
    ax2.set_ylabel('Google Trends (Lumber Price)', color='tab:orange')
    plt.title('WOOD Price vs. Google Trends (Lumber Price Search Volume)')
    fig.tight_layout()
    plt.show()

def time_series_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model, _ = train_gru(X_train, y_train, X_test, y_test)
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Fold {fold+1} Test Accuracy: {accuracy:.4f}")

def permutation_importance_gru(model, X_test, y_test, feature_names):
    baseline = model.evaluate(X_test, y_test, verbose=0)[1]  # accuracy
    importances = []
    for i in range(X_test.shape[2]):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, 0, i])
        score = model.evaluate(X_test_permuted, y_test, verbose=0)[1]
        importances.append(baseline - score)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(feature_names, importances)
    plt.title('Permutation Importance (GRU)')
    plt.ylabel('Decrease in Accuracy')
    plt.show()

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()
