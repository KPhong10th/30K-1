#fetch historical data and put into a csv from top tickers from industries
import yfinance as yf
import pandas as pd

# Define stock tickers (diversified set)
stocks = [
    'AAPL', 'MSFT', 'NVDA', 'AI',  # Technology
    'PFE', 'JNJ', 'MRNA',  # Healthcare
    'JPM', 'GS', 'V',  # Finance
    'TSLA', 'AMZN', 'HD',  # Consumer Discretionary
    'XOM', 'CVX', 'SHEL'  # Energy
]

# Define time period
start_date = '2024-6-01'
end_date = '2025-01-01'


def fetch_stock_data(tickers, start, end):
    """
    Fetches historical stock data for the specified tickers.
    Returns a DataFrame with closing prices.
    """
    all_data = []
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, start=start, end=end)['Close']
            if data.empty:
                print(f"No data for {ticker}, skipping.")
                continue
            # Convert Series to DataFrame and add a ticker column
            df = pd.DataFrame(data)
            df.rename(columns={'Close': ticker}, inplace=True)
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not all_data:
        raise ValueError("No valid data was fetched.")

    # Merge all dataframes on the date index
    combined_data = pd.concat(all_data, axis=1, join="inner")
    return combined_data


# Fetch and process data
try:
    stock_prices = fetch_stock_data(stocks, start_date, end_date)
    print("\nFirst few rows of the data:")
    print(stock_prices.head())
    print("\nData Information:")
    print(stock_prices.info())

    # Save to CSV
    stock_prices.to_csv("dec2024.csv")
    print("\nData successfully saved to 'dec2024.csv'.")
except Exception as e:
    print(f"An error occurred: {e}")
