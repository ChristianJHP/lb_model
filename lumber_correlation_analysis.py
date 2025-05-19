import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys

# List of tickers to analyze
tickers = [
    "LB=F",  # Lumber Futures
    "HD",    # Home Depot
    "LOW",   # Lowe's
    "KBH",   # KB Home
    "DHI",   # D.R. Horton
    "XLB",   # Materials Select Sector SPDR
    "PAVE",  # Global X U.S. Infrastructure
    "HG=F",  # Copper Futures
    "SLB",   # Steel
    "UNP",   # Union Pacific
    "JBHT",  # J.B. Hunt
    "RYN",   # Rayonier
    "PCH",   # PotlatchDeltic
    "^TYX"   # 30-Year Treasury Yield
]

def fetch_data():
    """Fetch historical data for all tickers"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    
    # Download data for each ticker separately to handle errors
    all_data = {}
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Columns for {ticker}: {data.columns}")
            print(data.head())
            if not data.empty:
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    if ('Close', ticker) in data.columns:
                        close_series = data[('Close', ticker)]
                    elif ('Adj Close', ticker) in data.columns:
                        close_series = data[('Adj Close', ticker)]
                    else:
                        print(f"Warning: No 'Close' or 'Adj Close' data for {ticker}")
                        continue
                else:
                    if 'Close' in data.columns:
                        close_series = data['Close']
                    elif 'Adj Close' in data.columns:
                        close_series = data['Adj Close']
                    else:
                        print(f"Warning: No 'Close' or 'Adj Close' data for {ticker}")
                        continue
                if isinstance(close_series, pd.Series) and close_series.notna().sum() > 0:
                    all_data[ticker] = close_series
                else:
                    print(f"Warning: No valid close data for {ticker}")
            else:
                print(f"Warning: No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
    
    # Combine all series into a DataFrame
    df = pd.DataFrame(all_data)
    return df

def calculate_correlations(data):
    """Calculate and display correlations"""
    if data.empty:
        print("Error: No data available for correlation analysis")
        sys.exit(1)
    
    # Calculate daily returns
    returns = data.pct_change()
    
    # Calculate correlation matrix using min_periods to allow for some missing data
    min_periods = int(len(returns) * 0.5)  # Require at least 50% of the data points
    corr_matrix = returns.corr(min_periods=min_periods)
    
    # Get top correlated assets with Lumber
    lumber_correlations = corr_matrix["LB=F"].sort_values(ascending=False)
    
    return corr_matrix, lumber_correlations

def plot_correlation_heatmap(corr_matrix):
    """Plot correlation heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt='.2f')
    plt.title("Correlation of Lumber Futures with Other Assets")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_top_correlations(lumber_correlations, top_n=5):
    # Exclude lumber itself
    lumber_correlations = lumber_correlations.drop('LB=F', errors='ignore')
    # Drop NaNs
    lumber_correlations = lumber_correlations.dropna()
    
    if lumber_correlations.empty:
        print("Warning: No valid correlations found to plot")
        return
        
    # Get top N positive and negative correlations
    top_pos = lumber_correlations.sort_values(ascending=False).head(top_n)
    top_neg = lumber_correlations.sort_values().head(top_n)
    top_combined = pd.concat([top_pos, top_neg])
    
    plt.figure(figsize=(8, 6))
    colors = ['red' if v < 0 else 'green' for v in top_combined]
    top_combined.sort_values().plot(kind='barh', color=colors)
    plt.title(f"Top {top_n} Positive and Negative Correlations with Lumber Futures")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.savefig('lumber_top_correlations.png')
    plt.close()

def main():
    # Fetch data
    data = fetch_data()
    
    if data.empty:
        print("Error: No data was downloaded. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Calculate correlations
    corr_matrix, lumber_correlations = calculate_correlations(data)
    
    # Plot heatmap
    plot_correlation_heatmap(corr_matrix)
    
    # Plot top correlations
    plot_top_correlations(lumber_correlations, top_n=5)
    
    # Print top correlations
    print("\nTop correlated assets with Lumber futures:")
    print(lumber_correlations)
    
    # Save correlations to CSV
    lumber_correlations.to_csv('lumber_correlations.csv')
    print("\nResults have been saved to 'lumber_correlations.csv', 'correlation_heatmap.png', and 'lumber_top_correlations.png'")

if __name__ == "__main__":
    main() 