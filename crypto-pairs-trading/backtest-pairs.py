import numpy as np
import pandas as pd
import pickle as pkl
import os

class Backtest:
    def __init__(self, df_close, initial_wealth, transaction_cost_type, transaction_cost,param_fitting_period,trading_period):
        """
        """
        self.asset_universe = list(df_close.columns) + ['Cash']  # Add cash as a "virtual asset"
        self.initial_wealth = initial_wealth
        self.transaction_cost_type = transaction_cost_type
        self.transaction_cost = transaction_cost
        self.prices = self.__trim_dataframe(df_close,n_non_nan=2)
        self.wealth = initial_wealth
        self.history = []
    

    def __trim_dataframe(df,n_non_nan=2):
        for i, row in df.iterrows():
            non_nan_count = row.notna().sum()
            if non_nan_count >= n_non_nan:
                return df.iloc[i:]
        
        return pd.DataFrame()

    def calculate_transaction_costs(self, target_weights):
        """
        Calculate the transaction cost for changing from current weights to target weights.
        
        :param target_weights: Array of target weights including cash allocation.
        :return: Transaction costs and net wealth impact.
        """
        weight_change = np.abs(target_weights - self.current_weights)
        asset_change = weight_change[:-1]
        cost = 0

        if self.transaction_cost_type == 'percentage':
            cost = self.transaction_cost * np.sum(asset_change) * self.wealth  # Exclude cash in transaction costs
        elif self.transaction_cost_type == 'absolute':
            cost = self.transaction_cost * (asset_change!=0).sum()

        return cost
    
    def execute_trade(self, target_weights):
        """
        Execute trades based on target weights and update the current portfolio weights and wealth.
        
        :param target_weights: Target portfolio weights to achieve, including cash.
        :return: Updated wealth and portfolio weights.
        """
        transaction_cost = self.calculate_transaction_costs(target_weights)
        self.wealth -= transaction_cost  # Subtract transaction cost from wealth

        # Calculate the portfolio return excluding the cash portion
        asset_return = np.dot(target_weights[:-1], self.get_current_returns())
        self.wealth *= (1 + asset_return)  # Update wealth based on portfolio returns (excluding cash)

        # Update current portfolio weights
        self.current_weights = target_weights

        return self.current_weights, self.wealth, transaction_cost
    
    def get_current_returns(self):
        """
        Get the returns of the asset universe for the current time step, excluding cash which has no return.
        
        :return: Array of returns for the current time step excluding cash.
        """
        return self.prices[self.prices.columns[:-1]].pct_change().iloc[self.current_timestep].values
    
    def _verify_rebalance(self,current_weights,target_weights,threshold=0.1):
        if (current_weights == np.zeros(len(current_weights))).all():
            return target_weights
        diff = target_weights - current_weights
        worth_changing = np.abs(diff) >= threshold
        up_wc = diff[worth_changing&(diff>0)].sum()
        down_wc = diff[worth_changing&(diff<0)].sum()
        up_tot = diff[diff>0].sum()
        down_tot = diff[diff<0].sum()
        if up_wc > -down_wc:
            diff[worth_changing&(diff>0)] = diff[worth_changing&(diff>0)]*(up_tot+(down_tot-down_wc))/up_wc
        else:
            diff[worth_changing&(diff<0)] = diff[worth_changing&(diff<0)]*(down_tot+(up_tot-up_wc))/down_wc
        diff[~worth_changing] = 0.0
        s = (current_weights + diff).sum()
        assert s>1-1e6 and s<1+1e6
        return current_weights + diff
    
    def calculate_weights(self, available_data):
        # """
        # Calculate target portfolio weights including cash allocation.
        # The user can customize this method to define their own strategy.

        # :param available_data: Historical price data up to the current time step.
        # :return: Array of target weights including cash.
        # """
        # n_assets = available_data.shape[1]
        # asset_weights = np.ones(n_assets - 1) / n_assets  # Equal weight allocation for assets
        # cash_weight = 0.10  # 10% allocation to cash

        # # Normalize the asset weights so the total allocation is 1
        # asset_weights = (1 - cash_weight) * asset_weights

        # # Combine the asset weights with the cash weight
        # target_weights = np.append(asset_weights, cash_weight)
        cash_weight = 0.0
        mean = self.predictions[self.current_timestep-1]['mean']
        cov = self.predictions[self.current_timestep-1]['cov']
        hrp_portfolio = HRPPortfolio(mean, cov)
        asset_weights = hrp_portfolio.get_hrp_weights()
        target_weights = np.append(asset_weights,cash_weight)
        target_weights = self._verify_rebalance(self.current_weights, target_weights)
        
        return target_weights
    
    def _update_wealth_weights(self, returns, weights, wealth):
        # Calculate the portfolio return excluding the cash portion
        new_weights = weights.copy()
        new_weights[:-1] = new_weights[:-1]*(1+returns)
        if new_weights[:-1].sum() != 0:
            new_weights[:-1] /= new_weights[:-1].sum()
        asset_returns = np.dot(weights[:-1],returns)
        new_wealth = wealth*(1+asset_returns)
        return new_weights, new_wealth
    
    def run(self):
        """
        Run the backtest over the specified date range, using the internal method to calculate asset weights.
        """
        for self.current_timestep in range(1, len(self.prices)):
            # Get the available price data up to the current timestep
            available_data = self.prices.iloc[:self.current_timestep]
            returns = self.get_current_returns()
            return_weights, return_wealth = self._update_wealth_weights(returns, self.current_weights, self.wealth)
            self.wealth = return_wealth
            self.current_weights = return_weights
            data = {
                'date': self.prices.index[self.current_timestep],
                'initial_wealth': return_wealth,
                'initial_weights': return_weights,
            }

            # Calculate target weights
            target_weights = self.calculate_weights(available_data)

            # Execute the trade based on target weights
            updated_weights, updated_wealth, transaction_cost = self.execute_trade(target_weights)
            data['end_wealth'] = updated_wealth
            data['end_weights'] = updated_weights
            data['transaction_cost'] = transaction_cost

            # Record the portfolio weights and wealth
            self.history.append(data)

    def get_results(self):
        """
        Get the results of the backtest as a Pandas DataFrame.
        
        :return: DataFrame containing the history of wealth and portfolio weights.
        """
        return pd.DataFrame(self.history)

if __name__=='__main__':
    # Example usage:
    asset_universe = ['Asset_A', 'Asset_B', 'Asset_C']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    initial_wealth = 2000
    transaction_cost_type = 'absolute'
    transaction_cost = 1

    # Create an instance of the Backtest class
    backtest = Backtest(asset_universe, start_date, end_date, initial_wealth, transaction_cost_type, transaction_cost)

    # Run the backtest
    backtest.run()

    # Retrieve and print the results
    results = backtest.get_results()
    df = pd.DataFrame(results)
    df = df.set_index('date',drop=True)
    df.to_csv(os.path.join(os.path.dirname(__file__),'results','montepred_hrp.csv'))
