# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:26:30 2023

@author: Shane
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root


def download_stock_data(stock_code, start_date, end_date):
    """
    Downloads and processes stock data for a given stock code.
    Calculates the monthly returns of the stock.

    Args:
    stock_code (str): Ticker symbol of the stock.
    start_date (str): Start date for the data download in 'YYYY-MM-DD' format.
    end_date (str): End date for the data download in 'YYYY-MM-DD' format.

    Returns:
    pandas.Series: Monthly returns of the stock.
    """
    data = yf.download(stock_code, start=start_date, end=end_date)
    monthly_data = data.resample('M').last()
    return monthly_data['Adj Close'].pct_change()

def build_returns_dataframe(stock_codes, start_date, end_date):
    """
    Builds a DataFrame containing monthly returns for a list of stocks.

    Args:
    stock_codes (list): List of stock ticker symbols.
    start_date (str): Start date for the data download.
    end_date (str): End date for the data download.

    Returns:
    pandas.DataFrame: DataFrame with each stock's monthly returns.
    """
    returns = {code: download_stock_data(code, start_date, end_date) for code in stock_codes}
    return pd.DataFrame(returns)

# List of stocks
stock_codes = ['VTSMX', 'FCNTX', 'AIVSX', 'PONAX', 'TRBCX', 'DODGX', 'MDLOX', 'FKGRX', 'OLGAX', 'PRNHX']

# Creating the DataFrame
monthly_returns_df = build_returns_dataframe(stock_codes, '2012-12-01', '2022-12-31')

# Dropping the first row to remove NaN values due to pct_change calculation
monthly_returns_df = monthly_returns_df.iloc[1:]

m_return = monthly_returns_df
'''
The code snippet under performs a financial optimization using a custom utility function defined as kink_1, and it uses scipy's minimize function for optimization. Let's walk through the code and discuss potential improvements for readability and maintainability:

kink_1 Function: This is a custom utility function. It applies a logarithmic transformation with an additional penalty for returns below -0.01. The function is clear and concise.

kink1_utility Function: This function calculates the utility of a portfolio based on the weights (params) of different stocks. It handles cases where the length of params is 9 or 10.

Optimization: The minimize function from scipy is used to find the optimal weights that maximize the utility function over a specified range of returns.

'''



def kink_1(x):
    """ Custom utility function with a kink at -0.01. """
    if x >= -0.01:
        return np.log(1 + x)
    else:
        return np.log(1 + x) + 10 * (x + 0.01)
   
def kink1_utility(portfolio_weights, returns_df, start_index, end_index):
    """
    Calculates the utility of a portfolio over a given range of returns.
    
    Args:
    portfolio_weights (list): Weights of the portfolio components.
    returns_df (DataFrame): DataFrame containing returns data.
    start_index (int): Start index for the calculation.
    end_index (int): End index for the calculation.

    Returns:
    float: Calculated utility value.
    """
    if len(portfolio_weights) == 9:
        total_weight = sum(portfolio_weights)
        if total_weight < 1:
            weights = portfolio_weights + [1 - total_weight]
        else:
            raise ValueError("Sum of weights exceeds 1")
    else:
        weights = portfolio_weights

    result = sum(kink_1(np.dot(np.array(weights), returns_df.iloc[i].values)) for i in range(start_index, end_index))
    return result / (end_index - start_index)



# Optimization
initial_guess = [0.1] * 10  # Adjusted to have 10 elements
initial_guess[-1] = 1 - sum(initial_guess[:-1])  # Adjust the last element so that the sum of weights is 1

'''
The bounds set between 0 and 1 for each weight, assuming that negative weights (short selling) are not allowed in this context. If short selling is allowed, you can adjust the bounds accordingly [-1,1].
'''
result = minimize(lambda weights: -kink1_utility(weights, monthly_returns_df, 0, 30), initial_guess, method='SLSQP', bounds=[(-1, 1)] * 10)

print("Optimal Weights:", result.x)
print("Optimized Function Value:", result.fun)


'''
Following code snippet that includes multiple utility functions (kink_5, S_0, S_5), their corresponding utility calculation functions (kink5_utility, s0_utility, s5_utility), and a Monte Carlo simulation to generate a portfolio's return, risk, and Sharpe ratio. The code also plots an efficient frontier graph with the market portfolio highlighted.
'''


def calculate_portfolio_return(weights, returns):
    """
    Calculate the expected return of a portfolio.

    Args:
    weights (np.array): Portfolio weights.
    returns (np.array): Expected returns for each asset.

    Returns:
    float: Portfolio expected return.
    """
    return np.dot(weights, returns)

def calculate_portfolio_risk(weights, cov_matrix):
    """
    Calculate the risk (standard deviation) of a portfolio.

    Args:
    weights (np.array): Portfolio weights.
    cov_matrix (np.array): Covariance matrix of the asset returns.

    Returns:
    float: Portfolio risk.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sharpe_ratio(portfolio_return, portfolio_risk, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio of a portfolio.

    Args:
    portfolio_return (float): Portfolio return.
    portfolio_risk (float): Portfolio risk.
    risk_free_rate (float): Risk-free rate of return.

    Returns:
    float: Sharpe Ratio.
    """
    return (portfolio_return - risk_free_rate) / portfolio_risk




def utility_function(x, type='kink5'):
    """
    Utility function for different portfolio returns.

    Args:
    x (float): Portfolio return.
    type (str): Type of utility function ('kink5', 's0', 's5').

    Returns:
    float: Calculated utility.
    """
    if type == 'kink5':
        if x >= -0.05:
            return np.log(1 + x)
        else:
            return np.log(1 + x) + 10 * (x + 0.01)
    elif type == 's0':
        return 2.25 * ((max(x, 0)) ** 0.01)
    elif type == 's5':
        return 2.25 * ((max(x - 0.005, 0)) ** 0.01)
'''

'''
def portfolio_utility(weights, returns, type='kink5'):
    """
    Calculate the utility of a portfolio.

    Args:
    weights (list): Portfolio weights.
    returns (pd.DataFrame): DataFrame containing returns data.
    type (str): Type of utility function.

    Returns:
    float: Average utility of the portfolio.
    """
    total_utility = 0
    for i in range(len(returns)):
        now_return = np.dot(np.array(weights), returns.iloc[i])
        total_utility += utility_function(now_return, type)

    return total_utility / len(returns)



def kink_5(x):
    if x >= -0.05:
        return np.log(1+x)
    else:
        return np.log(1+x) + 10*(x+0.01)
    
def kink5_utility(params,start_index,end_index):
    if len(params) == 9:
        w = list(params)
        w.append(1-sum(w))
    else:
        w = list(params)
    result = 0
    for i in range(start_index,end_index):
        now_return = np.dot(np.array(w),np.array(m_return.iloc[i]).reshape(10,1))[0]
        result += kink_5(now_return)
    return result/60

def S_0(x):
    if x > 0:
        return 2.25*((x - 0)**0.01)
    else:
        return -2.25*((0 - x)**0.01)
def s0_utility(params,start_index,end_index):
    if len(params) == 9:
        w = list(params)
        w.append(1-sum(w))
    else:
        w = list(params)
    result = 0
    for i in range(start_index,end_index):
        now_return = np.dot(np.array(w),np.array(m_return.iloc[i]).reshape(10,1))[0]
        result += S_0(now_return)
    return result/60

def S_5(x):
    if x > 0.005:
        return 2.25*((x - 0.005)**0.01)
    else:
        return -2.25*((0.005 - x)**0.01)
def s5_utility(params,start_index,end_index):
    if len(params) == 9:
        w = list(params)
        w.append(1-sum(w))
    else:
        w = list(params)
    result = 0
    for i in range(start_index,end_index):
        now_return = np.dot(np.array(w),np.array(m_return.iloc[i]).reshape(10,1))[0]
        result += S_5(now_return)
    return result/60




# Load your data here
# monthly_returns_df = ...

# Monte Carlo Simulation for Portfolio Optimization
num_simulations = 10000
expected_returns = np.mean(monthly_returns_df) * 12
cov_matrix = np.cov(monthly_returns_df.T)
num_assets = len(expected_returns)

risk, returns, sharpe_ratios = [], [], []

for _ in range(num_simulations):
    weights = np.random.uniform(-1, 1, size=num_assets)
    weights /= np.sum(weights)
    port_return = calculate_portfolio_return(weights, expected_returns)
    port_risk = calculate_portfolio_risk(weights, cov_matrix)
    risk.append(port_risk)
    returns.append(port_return)
    sharpe_ratios.append(sharpe_ratio(port_return, port_risk))

# Identifying the Market Portfolio
market_portfolio_index = np.argmax(sharpe_ratios)
market_risk = risk[market_portfolio_index]
market_return = returns[market_portfolio_index]
market_sharpe = sharpe_ratios[market_portfolio_index]

# Plotting the Efficient Frontier
plt.figure(figsize=(16, 8))
plt.scatter(risk, returns, c=sharpe_ratios, cmap='YlGnBu', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.title('Efficient Frontier with Market Portfolio')
plt.scatter(market_risk, market_return, c='red', marker='*', s=100, label='Market Portfolio')
plt.legend()
plt.ylim(-0.5, 0.5)
plt.xlim(0, 0.2)
plt.grid(True)
plt.show()


'MEAN VARIANCE ANALYSIS'

def MV_cal(target_return, start_index, end_index, returns_df):
    """
    Calculates the optimal weights for a portfolio to achieve a target return using Mean-Variance optimization.

    Args:
    target_return (float): The target return for the portfolio.
    start_index (int): The start index of the returns data for calculation.
    end_index (int): The end index of the returns data for calculation.
    returns_df (DataFrame): DataFrame containing the returns data.

    Returns:
    np.array: Optimal portfolio weights.
    """
    expected_returns = np.array(np.mean(returns_df.iloc[start_index:end_index])) * 12
    cov_matrix = np.cov(returns_df.iloc[start_index:end_index].T)

    cov_inverse = np.linalg.inv(cov_matrix)
    ones_vector = np.ones(len(expected_returns))

    U = -0.5 * np.dot(expected_returns, np.dot(cov_inverse, expected_returns))
    V = -0.5 * np.dot(expected_returns, np.dot(cov_inverse, ones_vector))
    W = -0.5 * np.dot(ones_vector, np.dot(cov_inverse, ones_vector))
    D = U * W - V * V

    lambda_value = (target_return * W - V) / D
    phi_value = -(target_return * V - U) / D

    optimal_weights = -0.5 * np.dot(cov_inverse, (lambda_value * expected_returns + phi_value * ones_vector))
    return optimal_weights

def utility_optimization(utility_func, start, end, returns, initial_guess):
    """
    Performs optimization using a specified utility function.

    Args:
    utility_func (function): The utility function to be maximized.
    start (int): Start index for the data.
    end (int): End index for the data.
    returns (DataFrame): DataFrame of returns.
    initial_guess (list): Initial guess for the optimization.

    Returns:
    OptimizeResult: The result of the optimization.
    """
    return minimize(lambda params: -utility_func(params, start, end, returns), 
                    initial_guess, method='SLSQP')

def calculate_portfolio_stats(weights, returns):
    """
    Calculates portfolio statistics like expected return, standard deviation, and Sharpe ratio.

    Args:
    weights (np.array): Portfolio weights.
    returns (DataFrame): DataFrame of returns.

    Returns:
    dict: Dictionary containing portfolio statistics.
    """
    expected_return = np.dot(weights, np.mean(returns) * 12)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))
    sharpe_ratio = expected_return / std_dev if std_dev != 0 else 0

    return {
        'expected_return': expected_return,
        'std_dev': std_dev,
        'sharpe_ratio': sharpe_ratio
    }

# Define your utility functions here: kink1_utility, kink5_utility, s0_utility, s5_utility

# Assuming m_return is your DataFrame with returns data
# m_return = 

# Optimization for different utility functions
utility_functions = [kink1_utility, kink5_utility, s0_utility, s5_utility]
utility_names = ['Kink 1%', 'Kink 5%', 'S_shape 0%', 'S_shape 0.5%']
initial_guess = [0.1] * 10  # Adjust number of elements as required

















