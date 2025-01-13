import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load stock data and calculate log returns
def load_data(file_path):
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def calculate_log_returns(data):
    return np.log(data / data.shift(1)).dropna()

# Portfolio optimization
def portfolio_optimization(mean_returns, cov_matrix, risk_free_rate, max_weight):
    num_assets = len(mean_returns)

    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)

    result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    portfolio_return = np.dot(result.x, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return {
        'weights': result.x,
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

# Scenario evaluation
def evaluate_scenarios(mean_returns, cov_matrix, scenarios):
    results = []

    for scenario in scenarios:
        risk_free_rate = scenario['risk_free_rate']
        max_weight = scenario['max_weight']

        result = portfolio_optimization(mean_returns, cov_matrix, risk_free_rate, max_weight)
        results.append({
            'risk_free_rate': risk_free_rate,
            'max_weight': max_weight,
            'weights': result['weights'],
            'return': result['return'],
            'volatility': result['volatility'],
            'sharpe_ratio': result['sharpe_ratio']
        })

    return pd.DataFrame(results)

# Main script
if __name__ == "__main__":
    # Load data
    file_path = "diversified_stock_data.csv"
    stock_prices = load_data(file_path)

    # Calculate log returns
    log_returns = calculate_log_returns(stock_prices)

    # Calculate mean returns and covariance matrix
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    # Define scenarios
    scenarios = [
        {'risk_free_rate': 0.01, 'max_weight': 1},
        {'risk_free_rate': 0.03, 'max_weight': 1},
        {'risk_free_rate': 0.05, 'max_weight': 1},
        {'risk_free_rate': 0.03, 'max_weight': 1},
        {'risk_free_rate': 0.03, 'max_weight': 1}
    ]

    # Evaluate scenarios
    results = evaluate_scenarios(mean_returns, cov_matrix, scenarios)

    # Display results
    print("\nScenario Results:")
    print(results)

    # Identify the best portfolio (highest Sharpe ratio)
    best_portfolio = results.loc[results['sharpe_ratio'].idxmax()]
    weights = best_portfolio['weights']
    print(weights)
    print("\nBest Portfolio:")
    print(best_portfolio)
