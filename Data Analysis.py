import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Constants
FILE_PATH = "diversified_stock_data.csv"
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "PFE": "Healthcare", "JNJ": "Healthcare", "MRNA": "Healthcare",
    "JPM": "Finance", "GS": "Finance", "V": "Finance",
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "XOM": "Energy", "CVX": "Energy", "SHEL": "Energy"
}

# Load saved data
def load_data(file_path):
    """
    Load historical stock price data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data with the index as dates.
    """
    try:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        exit(1)

# Calculate log returns
def calculate_log_returns(data):
    """
    Calculate daily log returns for the stock data.

    Parameters:
        data (pd.DataFrame): Historical stock prices.

    Returns:
        pd.DataFrame: Log returns of the stock prices.
    """
    return np.log(data / data.shift(1)).dropna()

# Visualize a heatmap
def plot_heatmap(data, title):
    """
    Plot a heatmap for the given data.

    Parameters:
        data (pd.DataFrame): Data for the heatmap.
        title (str): Title for the heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

# Plot explained variance by PCA components
def plot_pca_variance(explained_variance):
    """
    Plot the explained variance by PCA components.

    Parameters:
        explained_variance (np.array): Explained variance ratio of PCA components.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance * 100,
        tick_label=[f"PC{i}" for i in range(1, len(explained_variance) + 1)],
    )
    plt.title("Explained Variance by Principal Components")
    plt.ylabel("Percentage of Variance Explained")
    plt.xlabel("Principal Component")
    plt.show()

# Rolling volatility plot
def plot_rolling_volatility(log_returns, window=30):
    """
    Plot rolling volatility for each stock.

    Parameters:
        log_returns (pd.DataFrame): Log returns of stocks.
        window (int): Rolling window size (default=30).
    """
    rolling_volatility = log_returns.rolling(window=window).std()
    plt.figure(figsize=(12, 6))
    for column in rolling_volatility.columns:
        plt.plot(rolling_volatility.index, rolling_volatility[column], label=column)
    plt.title(f"{window}-Day Rolling Volatility")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.legend(loc="upper left", ncol=3)
    plt.grid()
    plt.show()

# Perform sector-based aggregation
def calculate_sector_stats(log_returns, sector_map):
    """
    Calculate sector-level mean and standard deviation of returns.

    Parameters:
        log_returns (pd.DataFrame): Log returns of stocks.
        sector_map (dict): Mapping of tickers to sectors.

    Returns:
        tuple: DataFrames of mean and standard deviation of returns by sector.
    """
    # Create a mapping DataFrame
    sector_df = pd.DataFrame({'Ticker': log_returns.columns})
    sector_df['Sector'] = sector_df['Ticker'].map(sector_map)

    if sector_df['Sector'].isnull().any():
        missing = sector_df.loc[sector_df['Sector'].isnull(), 'Ticker']
        raise ValueError(f"Missing sector mapping for: {missing.tolist()}")

    # Transpose log_returns for merging
    returns_with_sector = log_returns.T
    returns_with_sector['Ticker'] = returns_with_sector.index
    merged = returns_with_sector.merge(sector_df, on='Ticker')

    # Select only numeric columns for aggregation
    numeric_columns = merged.select_dtypes(include=[np.number])

    # Group by sector and calculate mean and std
    mean_stats = numeric_columns.groupby(merged['Sector']).mean().T
    std_stats = numeric_columns.groupby(merged['Sector']).std().T

    return mean_stats, std_stats

# Perform hierarchical clustering
def plot_dendrogram(correlation_matrix):
    """
    Plot a dendrogram for the correlation matrix.

    Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix of stocks.
    """
    linked = linkage(correlation_matrix, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=correlation_matrix.columns, orientation="top", distance_sort="descending")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Stocks")
    plt.ylabel("Distance")
    plt.show()

# Visualize sector-level mean returns
def plot_sector_mean_returns(mean_returns):
    mean_returns.mean().plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
    plt.title("Sector-Level Mean Returns")
    plt.ylabel("Mean Log Return")
    plt.xlabel("Sector")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Visualize sector-level volatility
def plot_sector_volatility(std_returns):
    std_returns.mean().plot(kind='bar', figsize=(10, 6), color='orange', edgecolor='black')
    plt.title("Sector-Level Volatility (Standard Deviation)")
    plt.ylabel("Volatility")
    plt.xlabel("Sector")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Main script
if __name__ == "__main__":
    # Load the data
    stock_prices = load_data(FILE_PATH)
    log_returns = calculate_log_returns(stock_prices)

    # Basic statistics
    mean_returns = log_returns.mean()
    volatility = log_returns.std()
    correlation_matrix = log_returns.corr()

    print("Mean Returns:")
    print(mean_returns)
    print("\nVolatility:")
    print(volatility)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualizations
    plot_heatmap(correlation_matrix, "Correlation Matrix Heatmap")
    log_returns.hist(bins=50, figsize=(12, 8))
    plt.suptitle("Distribution of Log Returns")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.show()

    # PCA analysis
    pca = PCA()
    pca.fit(log_returns)
    plot_pca_variance(pca.explained_variance_ratio_)

    # Rolling volatility
    plot_rolling_volatility(log_returns)

    # Annualized volatility
    annualized_volatility = volatility * np.sqrt(252)
    print("Annualized Volatility:")
    print(annualized_volatility)

    # Sector-level statistics
    mean_sector_returns, std_sector_returns = calculate_sector_stats(log_returns, SECTOR_MAP)

    print("\nSector-Level Mean Returns:")
    print(mean_sector_returns)
    print("\nSector-Level Volatility (Standard Deviation):")
    print(std_sector_returns)

    # Sector visualizations
    plot_sector_mean_returns(mean_sector_returns)
    plot_sector_volatility(std_sector_returns)

    # Hierarchical clustering
    plot_dendrogram(correlation_matrix)
