import pandas as pd
import yfinance as yf
import os

# Load biotech stock list from CSV
file_path = r"C:\Users\kevin\OneDrive\Desktop\Data Projects\biotech_stocks.csv"

# Verify file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file does not exist at the specified path: {file_path}")

# Read the CSV file
biotech_stocks = pd.read_csv(file_path)
print("Loaded DataFrame:")
print(biotech_stocks.head())

# Check for the required column
if "Ticker" not in biotech_stocks.columns:
    raise ValueError("Column 'Ticker' is missing from the file. Please ensure the CSV file has a 'Ticker' column.")

# Extract tickers and clean the data
tickers = biotech_stocks["Ticker"].dropna().str.strip().tolist()
if not tickers:
    raise ValueError("No valid tickers found in the 'Ticker' column. Please check your CSV file.")

print("Biotech stocks to analyze:", tickers)

# Create directory for storing data
os.makedirs("biotech_data", exist_ok=True)

# Define date range
start_date = "2020-01-01"
end_date = "2023-01-01"

# Fetch and save data
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    try:
        # Fetch historical data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.to_csv(f"biotech_data/{ticker}_historical_data.csv")
        print(f"Saved historical data for {ticker}")

        # Fetch options data
        stock_ticker = yf.Ticker(ticker)
        options_expirations = stock_ticker.options
        for expiration in options_expirations[:2]:  # Limit to first 2 expirations
            opt_chain = stock_ticker.option_chain(expiration)
            opt_chain.calls.to_csv(f"biotech_data/{ticker}_calls_{expiration}.csv")
            opt_chain.puts.to_csv(f"biotech_data/{ticker}_puts_{expiration}.csv")
            print(f"Saved options data for {ticker} (Expiration: {expiration})")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

print("Data acquisition for biotech stocks completed.")
