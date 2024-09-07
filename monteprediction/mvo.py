import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle as pkl
import os

class MVOPortfolio:
    def __init__(self, returns, cov_matrix, min_weight=-0.5, max_weight=0.3):
        """
        Initialize the Sharpe Ratio Portfolio class.
        
        :param returns: Pandas Series or array of expected returns for each asset.
        :param cov_matrix: Pandas DataFrame or array of expected covariance matrix for the assets.
        :param min_weight: Minimum weight for any asset.
        :param max_weight: Maximum weight for any asset.
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def sharpe_ratio(self, weights):
        """
        Calculate the negative Sharpe ratio (since we will minimize the objective).
        
        :param weights: Weights of the assets in the portfolio.
        :return: Negative Sharpe ratio (since optimization minimizes by default).
        """
        portfolio_return = np.dot(weights, self.returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - 0.0) / portfolio_volatility  # Assuming risk-free rate is 0
        return -sharpe_ratio  # Minimize negative Sharpe ratio

    def optimize_portfolio(self):
        """
        Optimize the portfolio to maximize Sharpe ratio with given constraints.
        
        :return: Optimal weights that maximize the Sharpe ratio.
        """
        num_assets = len(self.returns)
        initial_weights = np.ones(num_assets) / num_assets  # Start with equal weights

        # Constraints: sum of weights equals 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights should be 1
        ]

        # Bounds: constraints on individual weights
        bounds = [(self.min_weight, self.max_weight) for _ in range(num_assets)]

        result = minimize(self.sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x


if __name__=='__main__':
    with open(os.path.join(os.path.dirname(__file__),'data','predictions.pkl'),'rb') as f:
        predictions = pkl.load(f)
        expected_returns = predictions[-1]['mean']
        expected_cov_matrix = predictions[-1]['cov']
    portfolio_optimizer = MVOPortfolio(expected_returns, expected_cov_matrix, min_weight=-0.2, max_weight=0.4)

    optimal_weights = portfolio_optimizer.optimize_portfolio()
    print("Optimal Weights:")
    print(optimal_weights)
    print(optimal_weights.sum())
