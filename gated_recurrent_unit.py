# Improved Lumber Futures Trading Model with GRU Neural Network

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    # 5-day return
    features['Return_5d'] = df['WOOD'].pct_change(5)
    # RSI (14-day)
    features['RSI_14'] = RSIIndicator(df['WOOD'], window=14).rsi()
    # ATR (14-day) - use close for high/low if only close is available
    features['ATR_14'] = AverageTrueRange(
        high=df['WOOD'], low=df['WOOD'], close=df['WOOD'], window=14
    ).average_true_range()
    # Target (example: 5-day forward return direction)
    features['Target'] = np.where(df['WOOD'].pct_change(5).shift(-5) > 0, 1, 0)
    # Clean up
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model, history = train_gru(X_train, y_train, X_test, y_test)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')
    plot_trends_vs_price(df)

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

if __name__ == '__main__':
    main()
