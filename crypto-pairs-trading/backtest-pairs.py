import numpy as np
import pandas as pd
import pickle as pkl
import os

from scipy.stats import norm
import multiprocessing
import itertools
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

class Backtest:
    def __init__(self, df_close, initial_wealth, transaction_cost_type, transaction_cost,param_fitting_period,trading_period, freq=12*24*365):
        """
        """
        self.asset_universe = list(df_close.columns) + ['Cash']  # Add cash as a "virtual asset"
        self.assets = list(df_close.columns)
        self.initial_wealth = initial_wealth
        self.transaction_cost_type = transaction_cost_type
        self.transaction_cost = transaction_cost

        self.param_fitting_period = param_fitting_period
        self.trading_period = trading_period
        self.freq=freq # Periods in a year

        self.prices = self.__trim_dataframe(df_close,n_non_nan=2)
        self.prices.loc[:,'Cash'] = 0
        self.volatility = self.__calc_volatilty(self.prices)
        self.volatility.loc[:,'Cash'] = 0
        self.wealth = initial_wealth
        self.history = []
        self.current_weights = np.array([0 for _ in self.prices.columns])  # Start with no positions
        self.current_weights[-1] = 1

    def __calc_volatilty(self,df):
        returns = df.pct_change(fill_method=None)
        rolling_std = returns.rolling(window=self.param_fitting_period).std()
        ann_vol = rolling_std*np.sqrt(self.freq)
        return ann_vol
    
    def __trim_dataframe(self,df,n_non_nan=2):
        for i, row in df.iterrows():
            non_nan_count = row.notna().sum()
            if non_nan_count >= n_non_nan:
                return df.loc[i:]
        
        return pd.DataFrame()

    def calc_transaction_costs(self, target_weights):
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
        transaction_cost = self.calc_transaction_costs(target_weights)
        target_amounts = self.wealth*target_weights
        target_amounts[-1] -= transaction_cost
        self.wealth = target_amounts.sum()  # Subtract transaction cost from wealth

        # Update current portfolio weights
        self.current_weights = target_amounts/self.wealth

        return self.current_weights, self.wealth, transaction_cost
    
    def get_current_returns(self):
        """
        Get the returns of the asset universe for the current time step, excluding cash which has no return.
        
        :return: Array of returns for the current time step excluding cash.
        """
        return self.prices[self.prices.columns].pct_change(fill_method=None).iloc[self.current_timestep].values
    
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
    
    def calc_weights(self, available_data):
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
    
    def calc_pair_weights_step(self,params,prices,volatility,open_pair,current_weights):
        series01 = prices.copy()
        series02 = volatility.copy()
        # rates_filtered = series01.copy()
        # rates_filtered = rates_filtered[rates_filtered.columns[0]]
        # rates_filtered.iloc[:] = 0
        stock_y = params['Stock_y'] 
        stock_x = params['Stock_x']

        price_y = series01[stock_y]
        price_x = series01[stock_x]

        # Get the volatilities
        sigma_y = series02[stock_y]
        sigma_x = series02[stock_x]

        # Compute the log spread of the pair
        z_spread = np.log(price_y / price_x)

        # Compute the volatilities for the stock spread with the ATM vols
        sigmaRN_z = np.sqrt(sigma_y ** 2 + sigma_x ** 2 - 2 * sigma_y * sigma_x * params["rho_xy"])

        # proportion of days until the next maturity date in years
        # k = np.arange(len(z_spread), 0, -1) / self.freq ### Changed this
        k = 1

        # Help variable that is used a lot
        sigma_T = sigmaRN_z * np.sqrt(k)
        
        # Calculate d1 and d2 based on the formulas
        d1 = (z_spread + (0.5*sigmaRN_z**2) * k) / sigma_T ## Why add the rates filtered there? It doesn't appear in Appendix D: An Option to Exchange One Asset for Another
        d2 = d1 - sigma_T                                                            ## Why use absolute value for the z spread?

        # Calculate delta_y and delta_x using the norm.cdf function
        delta_y = norm.cdf(d1)
        delta_x = norm.cdf(d2)

        open_val = (1 - params["lambda_1"] - params["lambda_2"]) * sigmaRN_z
        trigger = z_spread > open_val
        if trigger and not open_pair:
            open_pair = True
            weights = np.array([delta_y,-delta_x])
        elif trigger and open_pair:
            weights = current_weights
        elif not trigger:
            open_pair = False
            weights = np.array([0,0])
        else:
            raise RuntimeError("This should never happen")
        return weights, open_pair
    
    def __get_assets_from_pairs(self, pairs_list):
        asset_list = []
        for p in pairs_list:
            if p[0] not in asset_list: asset_list.append(p[0])
            if p[1] not in asset_list: asset_list.append(p[1])
        return asset_list
    
    def calc_weights(self,params_dict,prices,volatility,open_pairs,assets_list=None):
        # if set()
        # TODO: do not overwrite pairs_weights_dict
        pairs_weights_dict = {p:None for p in params_dict.keys()}
        recalc_assets_list = False
        for p in params_dict.keys():
            if pd.isna(prices[p[0]]) or pd.isna(prices[p[1]]):
                del params_dict[p]
                recalc_assets_list = True
            pairs_weights_dict[p], open_pairs[p] = self.calc_pair_weights_step(params_dict[p],prices,volatility,open_pairs[p],pairs_weights_dict[p])
        if assets_list is None or not recalc_assets_list: assets_list = self.__get_assets_from_pairs(list(params_dict.keys()))
        weights_dict = {a:0 for a in assets_list}
        for p in pairs_weights_dict.keys():
            weights_dict[p[0]] += pairs_weights_dict[p][0]
            weights_dict[p[1]] += pairs_weights_dict[p][1]
        n_open_pairs = sum([1 for k in open_pairs.keys() if open_pairs[k]])
        weights_dict = {a: weights_dict[a]/n_open_pairs for a in assets_list}
        return weights_dict, open_pairs, params_dict, assets_list

    def _update_wealth_weights(self, returns, weights, wealth):
        # Calculate the portfolio return excluding the cash portion
        new_weights = weights.copy()
        new_weights = new_weights*(1+returns)
        new_weights /= new_weights.sum()
        asset_returns = np.dot(weights,returns)
        new_wealth = wealth*(1+asset_returns)
        return new_weights, new_wealth
    
    def __estimate_parameters(self,y, x, Delta_t=1):
        """
        Estimate the parameters of the continuous mispricing model.

        Args:
            y (ndarray): Prices of asset y.
            x (ndarray): Prices of asset x.
            Delta_t (float): Time interval between observations (usually 1 day)

        Returns:
            tuple: Estimated parameters (mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy).
        """

        # Calculate the logarithmic prices
        Y = np.log(y)
        X = np.log(x)

        # Calculate the differences and spread
        A = Y[1:] - Y[:-1]
        B = X[1:] - X[:-1]
        z = Y[:-1] - X[:-1]

        # Obtain the number of observations
        n = len(z)
        
        # Estimate lambda_1 and lambda_2
        denominator = Delta_t * np.sum((z - np.mean(z))**2)
        lambda_1 = 2 * np.sum((A - np.mean(A)) * (z - np.mean(z))) / denominator
        lambda_2 = 2 * np.sum((B - np.mean(B)) * (z - np.mean(z))) / denominator
        
        # Estimate sigma_y and sigma_x
        sigma_y = np.sqrt(np.sum(((A - np.mean(A)) - lambda_1 * Delta_t * (z - np.mean(z)))**2) / (n * Delta_t))
        sigma_x = np.sqrt(np.sum(((B - np.mean(B)) - lambda_2 * Delta_t * (z - np.mean(z)))**2) / (n * Delta_t))

        # Estimate mu_y and mu_x
        mu_y = np.sum(A / Delta_t + lambda_1 * z + 0.5 * sigma_y**2) / n
        mu_x = np.sum(B / Delta_t - lambda_2 * z + 0.5 * sigma_x**2) / n
        
        # Calculate Z_y and Z_x
        Z_y = (A - (mu_y - lambda_1 * z - 0.5 * sigma_y**2) * Delta_t) / (sigma_y * np.sqrt(Delta_t))
        Z_x = (B - (mu_x + lambda_2 * z - 0.5 * sigma_x**2) * Delta_t) / (sigma_x * np.sqrt(Delta_t))
        
        # Calculate rho_xy
        rho_xy = np.mean(Z_y * Z_x)
        
        return mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy
    
    def __process_pair(self,pair, available_data):
        stock_y, stock_x = pair

        series01 = available_data.iloc[-self.param_fitting_period:,:]

        prices1 = series01[stock_y]
        prices2 = series01[stock_x]

        mu_y, mu_x, lambda_1, lambda_2, sigma_y, sigma_x, rho_xy = self.__estimate_parameters(
            prices1.values, prices2.values, 1
        )

        stability_condition = lambda_1 + lambda_2 > 0 
        cointegration_condition = lambda_1 * lambda_2 < 0   # opposite signs

        if stability_condition and cointegration_condition:
            valid_parameters = {
                'Stock_y': stock_y,
                'Stock_x': stock_x,
                'Start_Date': prices1.index[1],
                'End_Date': prices2.index[-1],
                'mu_y': mu_y,
                'mu_x': mu_x,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'sigma_y': sigma_y,
                'sigma_x': sigma_x,
                'rho_xy': rho_xy
            }
        else:
            valid_parameters = None

        return valid_parameters

    def perform_parallel_processing(self, all_pairs, series01, window_size, step_size):
        # Determine the number of CPU cores
        num_cores = multiprocessing.cpu_count() #- 1
        # Perform parallel processing
        with Parallel(n_jobs=num_cores, verbose=10) as parallel:
            results = parallel(
                delayed(self.__process_pair)(pair, series01, window_size, step_size) for pair in all_pairs
            )

        # Flatten the results
        parameters_list = [param for sublist in results for param in sublist]

        # Create a DataFrame from the list of parameters
        pairs_df = pd.DataFrame(parameters_list)

        return pairs_df

    def _calc_params(self, available_data):
        clean_data = available_data.dropna(axis=1)
        all_pairs = list(itertools.combinations(clean_data.columns, 2))
        all_pairs += [(y,x) for x,y in all_pairs]
        
        all_params = {}
        
        for p in all_pairs:
            params = self.__process_pair(p,available_data)
            if params:
                all_params[p] = params
        
        return all_params

    def _verify_rebalance(self,current_weights,target_weights,threshold=0.001):
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
        assert s>1-1e-6 and s<1+1e-6
        return current_weights + diff

    def __get_target_weights_from_weights_dict(self, weights_dict):
        weights = []
        for a in self.assets:
            if a in weights_dict.keys():
                weights.append(weights_dict[a])
            else:
                weights.append(0)
        weights += [1 - sum(weights)]
        target_weights = self._verify_rebalance(self.current_weights, np.array(weights),threshold=.001)
        return target_weights

    def run(self):
        """
        Run the backtest over the specified date range, using the internal method to calculate asset weights.
        """
        for self.current_timestep in range(self.param_fitting_period, len(self.prices)):
            # Get the available price data up to the current timestep
            available_price_data = self.prices.iloc[:self.current_timestep,:]
            available_vol_data = self.volatility.iloc[:self.current_timestep,:]
            returns = self.get_current_returns()
            returns = np.nan_to_num(returns, 0)
            # TODO: Meter aquí logica de que pasa si no hay precio de uno
            return_weights, return_wealth = self._update_wealth_weights(returns, self.current_weights, self.wealth)
            self.wealth = return_wealth
            self.current_weights = return_weights
            data = {
                'date': self.prices.index[self.current_timestep],
                'initial_wealth': return_wealth,
                'initial_weights': return_weights,
            }

            if self.current_timestep % self.param_fitting_period == 0:
                tradeable_period = False
                params_dict = self._calc_params(available_price_data.iloc[-self.param_fitting_period:,:])
                target_weights = np.array([0 for _ in self.assets] + [1])
                if len(params_dict)>0:
                    tradeable_period = True
                    tradeable_assets = self.__get_assets_from_pairs(params_dict)
                    open_pairs = {p: False for p in params_dict.keys()}
                    # available_cash = {p: updated_wealth/len(params_dict) for p in params_dict.keys()}
                continue
            elif tradeable_period:
                weights_dict, open_pairs, params_dict, tradeable_assets = self.calc_weights(params_dict,available_price_data.iloc[-1,:],available_vol_data.iloc[-1,:],open_pairs,tradeable_assets)
                target_weights = self.__get_target_weights_from_weights_dict(weights_dict)
            else:
                assert (self.current_weights == np.array([0 for _ in self.assets] + [1])).all()
                target_weights = np.array([0 for _ in self.assets] + [1])
                
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
    BASE_PATH = os.path.abspath(os.path.join(__file__,'..','..'))
    DATA_PATH = os.path.join(BASE_PATH,'data','crypto-5-min','complete_close_price.csv')
    data = pd.read_csv(DATA_PATH,parse_dates=True)
    data['close_time'] = pd.to_datetime(data['close_time'])
    data = data.rename(columns={
        'close_time':'date'
    })
    data.index = data['date']
    data = data.drop('date',axis=1)


    # Example usage:
    initial_wealth = 2000
    transaction_cost_type = 'absolute'
    transaction_cost = 1
    param_fitting_period = 12*24*7
    trading_period = 12*24
    backtest_config = {
        'df_close':data,
        'initial_wealth':initial_wealth,
        'transaction_cost_type':transaction_cost_type,
        'transaction_cost':transaction_cost,
        'param_fitting_period':param_fitting_period,
        'trading_period':trading_period,
    }
    backtest = Backtest(**backtest_config)

    # Run the backtest
    backtest.run()

    # Retrieve and print the results
    results = backtest.get_results()
    df = pd.DataFrame(results)
    df = df.set_index('date',drop=True)
    df.to_csv(os.path.join(os.path.dirname(__file__),'results','montepred_hrp.csv'))
