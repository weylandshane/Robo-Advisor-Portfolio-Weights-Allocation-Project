#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:34:41 2023

@author: chenghaozhu
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root

stock_data = yf.download('VTSMX', start='2012-12-01', end='2022-12-31')
monthly_data = stock_data.resample('M').last()
monthly_returns = monthly_data['Adj Close'].pct_change()
m_return = pd.DataFrame(index = monthly_returns.index,columns = \
                        ['VTSMX','FCNTX','AIVSX','PONAX','TRBCX','DODGX','MDLOX','FKGRX','OLGAX','PRNHX'])
m_return['VTSMX'] = monthly_returns
for stock_code in ['FCNTX','AIVSX','PONAX','TRBCX','DODGX','MDLOX','FKGRX','OLGAX','PRNHX']:
    stock_data = yf.download(stock_code, start='2012-12-01', end='2022-12-31')
    monthly_data = stock_data.resample('M').last()
    monthly_returns = monthly_data['Adj Close'].pct_change()
    m_return[stock_code] = monthly_returns
    
m_return = m_return.iloc[1:]

# kinked logic

test_dataframe = pd.DataFrame(index = list(range(10)),columns = m_return.columns)
index_list = []
for i in range(10):
    index_list.append(m_return.index[12*i+11])
    test_dataframe.iloc[i] = np.sum(m_return[12*i:12*i+12])
test_dataframe.index = index_list

# kinked utility pointed at 1%

def kink_1(x):
    if x >= -0.01:
        return np.log(1+x)
    else:
        return np.log(1+x) + 10*(x+0.01)
def kink1_utility(params,start_index,end_index):
    if len(params) == 9:
        w = list(params)
        w.append(1-sum(w))
    else:
        w = list(params)
    result = 0
    for i in range(start_index,end_index):
        now_return = np.dot(np.array(w),np.array(m_return.iloc[i]).reshape(10,1))[0]
        result += kink_1(now_return)
    return result/60

# kinked utility pointed at 5%

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

# S-shaped utility function pointed at 0%

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

# S-shaped utility function pointed at 5%

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

# Efficient frontier

r = 0
expected_returns = np.array(np.mean(m_return))*12
cov_matrix = np.cov(m_return.T)

# Monte Carlo simulation to generate portfolio return, risk, and weights
risk = []  
returns = []
sharpe_ratio = []

for _ in range(10000):
    weights = np.random.uniform(-1, 1, size=len(expected_returns))
    weights /= np.sum(weights)
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk.append(port_risk)
    returns.append(port_return)
    sharpe_ratio.append((port_return - r) / port_risk)

# market portfolio
market_risk = risk[np.argmax(sharpe_ratio)]
market_return = returns[np.argmax(sharpe_ratio)]
market_sharpe = max(sharpe_ratio)

# efficient frontier
plt.figure(figsize = (16,8))
plt.scatter(risk, returns, c=sharpe_ratio, cmap='YlGnBu', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.title('Efficient Frontier with Market Portfolio')
plt.scatter(market_risk, market_return, c='red', marker='*', s=100, label='Market Portfolio')
plt.legend()
plt.ylim(-0.5,0.5)
plt.xlim(0,0.2)
plt.grid(True)
plt.show()

# optimial MV weight

def MV_cal(target_miu,start_index,end_index):
    expected_returns = np.array(np.mean(m_return.iloc[start_index:end_index]))*12
    cov_matrix = np.cov(m_return.iloc[start_index:end_index].T)
    cov_inverse = np.linalg.inv(cov_matrix)
    vec_1 = np.array([1,1,1,1,1,1,1,1,1,1])
    U = -0.5*np.dot(expected_returns,np.dot(cov_inverse,expected_returns.reshape(10,1)))[0]
    V = -0.5*np.dot(expected_returns,np.dot(cov_inverse,vec_1.reshape(10,1)))[0]
    W = -0.5*np.dot(vec_1,np.dot(cov_inverse,vec_1.reshape(10,1)))[0]
    D = U*W - V*V
    lamda = (target_miu*W - V)/D
    phi = -(target_miu*V - U)/D
    w_result = -0.5*np.dot(cov_inverse,(lamda*expected_returns + phi*vec_1))
    return w_result

# Comparison between F-S and M-V

for i in range(3):
    expected_returns = np.array(np.mean(m_return.iloc[i*30:i*30+30]))*12
    in_sample_result = pd.DataFrame(index = ['Expected return','std','Expected utility','weights'],\
                                    columns = ['Kink 1% FS','Kink 1% MV','Kink 5% FS','Kink 5% MV',\
                                               'S_shape 0% FS','S_shape 0% MV','S_shape 0.5% FS','S_shape 0.5% MV'])
    initial_guess = [0.1] * 9  
    result = minimize(lambda params: -kink1_utility(params,i*30,i*30+30), initial_guess, method='SLSQP')
    weights = list(result.x)
    weights.append(1 - sum(result.x))
    weights = np.array(weights)
    in_sample_result['Kink 1% FS'].loc['weights'] = weights
    in_sample_result['Kink 1% FS'].loc['Expected return'] = np.dot(weights, expected_returns)
    in_sample_result['Kink 1% FS'].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights)))
    in_sample_result['Kink 1% FS'].loc['Expected utility'] = kink1_utility(weights,i*30,i*30+30)
    in_sample_result['Kink 1% MV'].loc['Expected return'] = np.dot(weights, expected_returns)
    weights_MV = MV_cal(in_sample_result['Kink 1% MV'].loc['Expected return'],i*30,i*30+30)
    in_sample_result['Kink 1% MV'].loc['weights'] = weights_MV
    in_sample_result['Kink 1% MV'].loc['std'] = np.sqrt(np.dot(weights_MV.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights_MV)))
    in_sample_result['Kink 1% MV'].loc['Expected utility'] = kink1_utility(weights_MV,i*30,i*30+30)
    initial_guess = [0.1] * 9  
    result = minimize(lambda params: -kink5_utility(params,i*30,i*30+30), initial_guess, method='SLSQP')
    weights = list(result.x)
    weights.append(1 - sum(result.x))
    weights = np.array(weights)
    in_sample_result['Kink 5% FS'].loc['weights'] = weights
    in_sample_result['Kink 5% FS'].loc['Expected return'] = np.dot(weights, expected_returns)
    in_sample_result['Kink 5% FS'].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights)))
    in_sample_result['Kink 5% FS'].loc['Expected utility'] = kink5_utility(weights,i*30,i*30+30)
    in_sample_result['Kink 5% MV'].loc['Expected return'] = np.dot(weights, expected_returns)
    weights_MV = MV_cal(in_sample_result['Kink 5% MV'].loc['Expected return'],i*30,i*30+30)
    in_sample_result['Kink 5% MV'].loc['weights'] = weights_MV
    in_sample_result['Kink 5% MV'].loc['std'] = np.sqrt(np.dot(weights_MV.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights_MV)))
    in_sample_result['Kink 5% MV'].loc['Expected utility'] = kink5_utility(weights_MV,i*30,i*30+30)
    initial_guess = [0.1] * 9  
    result = minimize(lambda params: -s0_utility(params,i*30,i*30+30), initial_guess, method='SLSQP', bounds=[(-1, 1)] * 9)
    weights = list(result.x)
    weights.append(1 - sum(result.x))
    weights = np.array(weights)
    in_sample_result['S_shape 0% FS'].loc['weights'] = weights
    in_sample_result['S_shape 0% FS'].loc['Expected return'] = np.dot(weights, expected_returns)
    in_sample_result['S_shape 0% FS'].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights)))
    in_sample_result['S_shape 0% FS'].loc['Expected utility'] = s0_utility(weights,i*30,i*30+30)
    in_sample_result['S_shape 0% MV'].loc['Expected return'] = np.dot(weights, expected_returns)
    weights_MV = MV_cal(in_sample_result['S_shape 0% MV'].loc['Expected return'],i*30,i*30+30)
    in_sample_result['S_shape 0% MV'].loc['weights'] = weights_MV
    in_sample_result['S_shape 0% MV'].loc['std'] = np.sqrt(np.dot(weights_MV.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights_MV)))
    in_sample_result['S_shape 0% MV'].loc['Expected utility'] = s0_utility(weights_MV,i*30,i*30+30)
    initial_guess = [0.1] * 9  
    result = minimize(lambda params: -s5_utility(params,i*30,i*30+30), initial_guess, method='SLSQP', bounds=[(-1, 1)] * 9)
    weights = list(result.x)
    weights.append(1 - sum(result.x))
    weights = np.array(weights)
    in_sample_result['S_shape 0.5% FS'].loc['weights'] = weights
    in_sample_result['S_shape 0.5% FS'].loc['Expected return'] = np.dot(weights, expected_returns)
    in_sample_result['S_shape 0.5% FS'].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights)))
    in_sample_result['S_shape 0.5% FS'].loc['Expected utility'] = s5_utility(weights,i*30,i*30+30)
    in_sample_result['S_shape 0.5% MV'].loc['Expected return'] = np.dot(weights, expected_returns)
    weights_MV = MV_cal(in_sample_result['S_shape 0.5% MV'].loc['Expected return'],i*30,i*30+30)
    in_sample_result['S_shape 0.5% MV'].loc['weights'] = weights_MV
    in_sample_result['S_shape 0.5% MV'].loc['std'] = np.sqrt(np.dot(weights_MV.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights_MV)))
    in_sample_result['S_shape 0.5% MV'].loc['Expected utility'] = s5_utility(weights_MV,i*30,i*30+30)
    print("In Sample")
    print(in_sample_result)
    out_sample_result = pd.DataFrame(index = ['Expected return','std','Sharpe ratio','Expected utility','weights'],\
                                    columns = ['Kink 1% FS','Kink 1% MV','Kink 5% FS','Kink 5% MV',\
                                               'S_shape 0% FS','S_shape 0% MV','S_shape 0.5% FS','S_shape 0.5% MV'])
    expected_returns = np.array(np.mean(m_return.iloc[i*30+30:i*30+60]))*12
    for a in ['Kink 1% FS','Kink 1% MV']:
        weights = in_sample_result[a].loc['weights']
        out_sample_result[a].loc['weights'] = weights
        out_sample_result[a].loc['Expected return'] = np.dot(weights, expected_returns)
        out_sample_result[a].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30+30:i*30+60].T), weights)))
        out_sample_result[a].loc['Sharpe ratio'] = out_sample_result[a].loc['Expected return']/out_sample_result[a].loc['std']
        out_sample_result[a].loc['Expected utility'] = kink1_utility(weights,i*30+30,i*30+60)
    for a in ['Kink 5% FS','Kink 5% MV']:
        weights = in_sample_result[a].loc['weights']
        out_sample_result[a].loc['weights'] = weights
        out_sample_result[a].loc['Expected return'] = np.dot(weights, expected_returns)
        out_sample_result[a].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30+30:i*30+60].T), weights)))
        out_sample_result[a].loc['Sharpe ratio'] = out_sample_result[a].loc['Expected return']/out_sample_result[a].loc['std']
        out_sample_result[a].loc['Expected utility'] = kink5_utility(weights,i*30+30,i*30+60)
    for a in ['S_shape 0% FS','S_shape 0% MV']:
        weights = in_sample_result[a].loc['weights']
        out_sample_result[a].loc['weights'] = weights
        out_sample_result[a].loc['Expected return'] = np.dot(weights, expected_returns)
        out_sample_result[a].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30+30:i*30+60].T), weights)))
        out_sample_result[a].loc['Sharpe ratio'] = out_sample_result[a].loc['Expected return']/out_sample_result[a].loc['std']
        out_sample_result[a].loc['Expected utility'] = s0_utility(weights,i*30+30,i*30+60)
    for a in ['S_shape 0.5% FS','S_shape 0.5% MV']:
        weights = in_sample_result[a].loc['weights']
        out_sample_result[a].loc['weights'] = weights
        out_sample_result[a].loc['Expected return'] = np.dot(weights, expected_returns)
        out_sample_result[a].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30+30:i*30+60].T), weights)))
        out_sample_result[a].loc['Sharpe ratio'] = out_sample_result[a].loc['Expected return']/out_sample_result[a].loc['std']
        out_sample_result[a].loc['Expected utility'] = s5_utility(weights,i*30+30,i*30+60)
    print("Out sample")
    print(out_sample_result)

# weights for risk parity

def RP_cal(alpha,start_index,end_index):
    w_list = []
    for code in m_return.columns:
        er = np.mean(m_return.iloc[start_index:end_index][code])
        sigma = np.std(m_return.iloc[start_index:end_index][code])
        w_list.append(((1+er)**alpha)/sigma)
    k = 1/(sum(w_list))
    w_array = np.array(w_list)
    w_array *= k
    return w_array

# Comparison between risk parity and M-V

for i in range(3):
    in_sample_rp = pd.DataFrame(index = ['Expected return','std','Sharpe ratio','weights'],\
                                    columns = ['RP 0','MV 0','RP 10','MV 10','RP -10','MV -10'])
    expected_returns = np.array(np.mean(m_return.iloc[i*30:i*30+30]))*12
    for alpha in [0,10,-10]:
        weights = RP_cal(alpha,i*30,i*30+30)
        rp_column = 'RP ' + str(alpha)
        mv_column = 'MV ' + str(alpha)
        in_sample_rp[rp_column].loc['weights'] = weights
        in_sample_rp[rp_column].loc['Expected return'] = np.dot(weights, expected_returns)
        in_sample_rp[rp_column].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights)))
        in_sample_rp[rp_column].loc['Sharpe ratio'] = in_sample_rp[rp_column].loc['Expected return']/in_sample_rp[rp_column].loc['std']
        in_sample_rp[mv_column].loc['Expected return'] = in_sample_rp[rp_column].loc['Expected return']
        weights_MV = MV_cal(in_sample_rp[mv_column].loc['Expected return'],i*30,i*30+30)
        in_sample_rp[mv_column].loc['weights'] = weights_MV
        in_sample_rp[mv_column].loc['std'] = np.sqrt(np.dot(weights_MV.T, np.dot(np.cov(m_return.iloc[i*30:i*30+30].T), weights_MV)))
        in_sample_rp[mv_column].loc['Sharpe ratio'] = in_sample_rp[mv_column].loc['Expected return']/in_sample_rp[mv_column].loc['std']
    print("In Sample")
    print(in_sample_rp)
    out_sample_rp = pd.DataFrame(index = ['Expected return','std','Sharpe ratio','weights'],\
                                    columns = ['RP 0','MV 0','RP 10','MV 10','RP -10','MV -10'])
    expected_returns = np.array(np.mean(m_return.iloc[i*30+30:i*30+60]))*12
    for a in ['RP 0','MV 0','RP 10','MV 10','RP -10','MV -10']:
        weights = in_sample_rp[a].loc['weights']
        out_sample_rp[a].loc['weights'] = weights
        out_sample_rp[a].loc['Expected return'] = np.dot(weights, expected_returns)
        out_sample_rp[a].loc['std'] = np.sqrt(np.dot(weights.T, np.dot(np.cov(m_return.iloc[i*30+30:i*30+60].T), weights)))
        out_sample_rp[a].loc['Sharpe ratio'] = out_sample_rp[a].loc['Expected return']/out_sample_rp[a].loc['std']
    print("Out Sample")
    print(out_sample_rp)
