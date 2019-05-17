#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import talib

#from talib import abstract
#from talib.abstract import *
# In[1]:


def load_data(path):
    market_data = pd.read_csv(path)
    market_data.date = pd.to_datetime(market_data.date)
    market_data['month'] = np.sin(market_data.date.dt.month*2*math.pi/12)
    market_data['day'] = np.sin(market_data.date.dt.day*2*math.pi/31)
    market_data['dayofweek'] = np.sin(market_data.date.dt.dayofweek*2*math.pi/5)
    market_data = market_data.drop('year', axis=1)
    market_data['sell'] = 0
    market_data['hold'] = 1
    market_data['buy'] = 0
    market_data['profit'] = 0
    market_data['allocation'] = 0
    
    return market_data

# In[5]:
def mkdir(exp_result_path, project_name, assets, sub_folder):
    if not (os.path.isdir('%s/%s'%(exp_result_path, project_name))):
        os.mkdir('%s/%s'%(exp_result_path, project_name))
    for asset in assets:
        if not (os.path.isdir('%s/%s/%s'%(exp_result_path, project_name, asset))):
            os.mkdir('%s/%s/%s'%(exp_result_path, project_name, asset))
    for asset in assets:
        for folder in sub_folder:
            if not (os.path.isdir('%s/%s/%s/%s'%(exp_result_path, project_name, asset, folder))):
                os.mkdir('%s/%s/%s/%s'%(exp_result_path, project_name, asset, folder))

def create_folder(assets, fname):
    if not (os.path.isdir('exp_result')):
        os.mkdir('exp_result')
    for asset in assets:
        if not (os.path.isdir('exp_result//'+asset)):
            os.mkdir('exp_result//'+asset)
    for asset in assets:
        if not (os.path.isdir('exp_result//%s//%s'%(asset, fname))):
            os.mkdir('exp_result//%s//%s'%(asset, fname))
            
def load_stock_data(asset, train_asset, valid_asset):
    train_asset_1 = train_asset[train_asset.assetName==asset].fillna(0).reset_index(drop=True)
    valid_asset_1 = valid_asset[valid_asset.assetName==asset].fillna(0).reset_index(drop=True)
    buy_hold = valid_asset_1.tail(250).reset_index(drop=True)
    buy_hold = (buy_hold.close / buy_hold.close[0])
    return train_asset_1, valid_asset_1, buy_hold

def TI(train_asset_1):
    cols = ['open', 'high', 'low', 'close', 'volume']
    inputs = {i:train_asset_1[i].values for i in cols}
    #WILLR = abstract.Function('WILLR')
    #STOCH = abstract.Function('STOCH')
    #ROCP = abstract.Function('ROCP')
    train_asset_1['SMA'] = SMA(inputs, timeperiod=15, price='close')
    macd = MACD(inputs)
    train_asset_1['macd'] = macd[0]
    train_asset_1['macdsignal'] = macd[1]
    train_asset_1['macdhist'] = macd[2]

    train_asset_1['RSI'] = RSI(inputs)
    train_asset_1['WILLR'] = WILLR(inputs)
    stoch = STOCH(inputs, fastk_period=9)
    train_asset_1['STOCH_D'] = stoch[0]
    train_asset_1['STOCH_K'] = stoch[1]
    train_asset_1['ROCP'] = ROCP(inputs)
    train_asset_1 = train_asset_1.fillna(0)
    return train_asset_1


def set_state(data, n_state, action):
    if n_state > data.shape[0]-1:
        return data
    arr = [0,0,0]
    arr[action] = 1
    data.loc[n_state, ['sell', 'hold', 'buy']] = arr
    return data

def set_something(data, n_state, col, V):
    data.loc[n_state, col] = V
    return data

def get_action_position(action, position):
    if action == position:
        action = 1
    if position == 1:
        position = action
    elif abs(position-action)==2:
        position = 1
    else:
        position = position
    return action, position

def check_ended(stock_trader, train_asset_1, n_state):
    if stock_trader.is_holding_stock:
        stock_trader.sell(train_asset_1.loc[n_state, 'close'])
    elif stock_trader.is_shorting_stock:
        stock_trader.buy(train_asset_1.loc[n_state, 'close'])
    return stock_trader

def action_pie(a_TJ):
    a_0 = (a_TJ==0).sum()
    a_1 = (a_TJ==1).sum()
    a_2 = (a_TJ==2).sum()

    plt.pie([a_0,a_1,a_2] , labels = [0,1,2],autopct='%1.1f%%', colors=['g', 'c', 'r'])
    plt.axis('equal')
    plt.show()
    
def action_scatter(a_TJ, valid_asset_1):
    buy_sig = pd.DataFrame((a_TJ==2).astype(int))
    buy_sig = buy_sig[buy_sig.values==1].index
    
    sel_sig = pd.DataFrame((a_TJ==0).astype(int))
    sel_sig = sel_sig[sel_sig.values==1].index
    
    plt.scatter(buy_sig, valid_asset_1['close'][buy_sig], s=10, c='r', alpha=1)
    plt.scatter(sel_sig, valid_asset_1['close'][sel_sig], s=10, c='g', alpha=1)
    
    plt.plot(valid_asset_1['close'], 'b', alpha=0.5)
    plt.show()
    
def profit2gather(asset_TJ, valid_asset_1):
    Total_profit = asset_TJ[-1]/ valid_asset_1.close[0]
    plt.plot(asset_TJ/ valid_asset_1.close[0], 'r')
    plt.plot(buy_hold, 'g')
    plt.show()
    print ('buy_hold profit', buy_hold.iloc[-1])
    print ('Total profit', '\033[1;31m '+str(Total_profit[0])+' \033[0m')
    print ('Beat B&H stategy: ', '\033[1;31m '+str(Total_profit[0]- buy_hold.iloc[-1])+' \033[0m')






