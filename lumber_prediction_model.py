from pytrends.request import TrendReq

def fetch_google_trends(keyword, start_date, end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [keyword]
    pytrends.build_payload(kw_list, cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    trends = pytrends.interest_over_time()
    if 'isPartial' in trends.columns:
        trends = trends.drop(columns=['isPartial'])
    trends = trends.rename(columns={keyword: 'GoogleTrends_Lumber'})
    return trends

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
    trends = fetch_google_trends('lumber price', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = df.merge(trends, left_index=True, right_index=True, how='left')
    df['GoogleTrends_Lumber'] = df['GoogleTrends_Lumber'].ffill()
    df = df.dropna()
    return df

def add_features(df):
    features = pd.DataFrame(index=df.index)
    # 5-day return
    features['Return_5d'] = df['WOOD'].pct_change(5)
    # RSI (14-day)
    features['RSI_14'] = RSIIndicator(df['WOOD'], window=14).rsi()
    # ATR (14-day)
    features['ATR_14'] = AverageTrueRange(
        high=df['WOOD'], low=df['WOOD'], close=df['WOOD'], window=14
    ).average_true_range()
    # Google Trends
    features['GoogleTrends_Lumber'] = df['GoogleTrends_Lumber']
    # Target (example: 5-day forward return direction)
    features['Target'] = np.where(df['WOOD'].pct_change(5).shift(-5) > 0, 1, 0)
    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features 