import numpy as np
import pandas as pd
import pickle as pkl
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import os

class HRPPortfolio:
    def __init__(self, returns, cov_matrix,asset_names=None):
        """
        Initialize the HRP portfolio class.
        
        :param returns: Pandas Series or array of expected returns for each asset.
        :param cov_matrix: Pandas DataFrame or array of expected covariance matrix for the assets.
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.dist_matrix = self.get_correlation_distances()
        if asset_names is None:
            asset_names = ['asset_'+str(i) for i in range(len(returns))]
        self.asset_names = asset_names

    def get_correlation_distances(self):
        covariance = self.cov_matrix
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        dist_matrix = np.sqrt(0.5 * (1 - correlation))
        np.fill_diagonal(dist_matrix, 0)
        return squareform(dist_matrix)
    
    def cluster_assets(self):
        """
        Perform hierarchical clustering on the distance matrix of the assets.
        
        :return: Cluster linkage matrix.
        """
        return linkage(self.dist_matrix, method='ward')

    def get_quasi_diag(self, linkage_matrix):
        """
        Get the order of assets according to the hierarchical tree.
        
        :param linkage_matrix: Linkage matrix from clustering.
        :return: Ordered list of asset indices.
        """
        return self._get_recursive_order(linkage_matrix, len(linkage_matrix) + 1)

    def _get_recursive_order(self, linkage_matrix, num_assets):
        """
        Recursively extract the order of assets from the linkage matrix.
        
        :param linkage_matrix: Linkage matrix from clustering.
        :param num_assets: Number of assets in the portfolio.
        :return: Ordered list of asset indices.
        """
        if num_assets == 1:
            return [num_assets - 1]
        
        leaf_order = list(range(num_assets))
        clusters = dendrogram(linkage_matrix, no_plot=True)['leaves']
        
        return [leaf_order[i] for i in clusters]

    def inverse_variance_portfolio(self, ordered_assets):
        """
        Compute the inverse variance portfolio based on the ordered assets.
        
        :param ordered_assets: List of asset indices ordered by hierarchical clustering.
        :return: Pandas Series of target weights.
        """
        inv_var = 1 / np.diag(self.cov_matrix).take(ordered_assets)
        target_weights = inv_var / np.sum(inv_var)
        # return pd.Series(target_weights, index=self.cov_matrix.index[ordered_assets])
        return target_weights

    def get_hrp_weights(self):
        """
        Compute the final target weights using Hierarchical Risk Parity (HRP).
        
        :return: Pandas Series of HRP target weights.
        """
        linkage_matrix = self.cluster_assets()  # Step 1: Perform clustering
        ordered_assets = self.get_quasi_diag(linkage_matrix)  # Step 2: Get quasi-diagonal order
        hrp_weights = self.inverse_variance_portfolio(ordered_assets)  # Step 3: Compute inverse variance weights
        return hrp_weights

if __name__=='__main__':
    with open(os.path.join(os.path.dirname(__file__),'data','predictions.pkl'),'rb') as f:
        predictions = pkl.load(f)
        expected_returns = predictions[-1]['mean']
        expected_cov_matrix = predictions[-1]['cov']
    hrp_portfolio = HRPPortfolio(expected_returns, expected_cov_matrix)

    target_weights = hrp_portfolio.get_hrp_weights()
    print("HRP Target Weights:")
    print(target_weights)
    print(target_weights.sum())
