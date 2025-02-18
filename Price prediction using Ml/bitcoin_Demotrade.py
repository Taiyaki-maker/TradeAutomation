# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sympy
import math
import time

#0.01
r1 = 0.01
#0.004
r2 = 0.004

df = pd.read_csv("C:\\Users\\macka\\.spyder-py3\\開発したアプリ\\使用したデータセット一覧\\テストデータver4.0.csv", header=None, encoding='utf-8')
df.tail()
X = np.array(df.iloc[0:12250,2].values)
X = X.astype(float)

df2 = pd.read_csv("C:\\Users\\macka\\.spyder-py3\\開発したアプリ\\使用したデータセット一覧\\テストデータver2.0.csv", header=None, encoding='utf-8')
df2.tail()
df2 = df2.iloc[::-1]
df2.index = range(len(df2))
Xt = np.array(df2.iloc[0:7,2].values)
Xt = Xt.astype(float)
yt = np.array([1 if df2.iloc[7,2] > df2.iloc[6,2] else -1])

def predict(y,x):
    y2 = []
    for i in x:
        y2.append(y[i])
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y2)
    mod = LinearRegression()
    lr = mod.fit(df_x, df_y)
    return lr.coef_[0][0],lr.intercept_[0]

def choice(short,long):
    ε = 30
    ε2 = 80
    ε3 = 300
    grad = 90
    flag = False
    data = np.arange(985, 1000, 1)
    a1,b1 = predict(short,data)
    a2,b2 = predict(long,data)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    ex1 = -y + a1*x + b1
    ex2 = -y + a2*x + b2
    dict = sympy.solve([ex1,ex2])
    #print(a1 > 100, a1 < -100 ,dict[x] - 999 < ε ,dict[x] >= 1000 ,abs(short[999] - long[999]) < ε2)
    #print(dict[x])
    if (abs(short[999] - long[999]) < ε2 or flag or a1 > grad or a1 < -grad or dict[x] - 999 < ε and dict[x] >= 1000) and abs(short[999] - long[999]) < ε3:
        if short[999] > long[999] and a1 < 0:
            return -1
        elif short[999] < long[999] and a1 > 0:
            return 1
    return 0

def possetion(trade_info,now_price,jpy):
    output = 0
    if trade_info:
        output = trade_info[0][1]
    return  output * now_price + jpy

i = 0
jpy = 1000000
last_price = X[0]
trade_info = []
hikaku = []
hikaku2 = []
count = 0
data = {}

buy_x = []
buy_y = []
sell_x = []
sell_y = []


for i in range(1000,len(X),1):
    Y = X[i-1000:i]
    df = pd.DataFrame(Y)
    #指数移動平均線を描画
    short = df[0].ewm(span=48, adjust=False).mean()
    long = df[0].ewm(span=288, adjust=False).mean()
    now_price = X[i]
    x = np.arange(i-1000,i, 1)
    
    plt.plot(x, short, 'r-')
    plt.plot(x, long, 'b-')
    plt.plot(buy_x,buy_y,"o", markersize = 8, color = 'g')
    plt.plot(sell_x,sell_y,"o", markersize = 8, color = 'y')
    plt.show()
    buy_amount = jpy / now_price
    #print(choice(short,long),i)
    if choice(short,long) == 1 and int(jpy) > 0:
        jpy -= buy_amount * now_price
        print("購入しました",i)
        buy_x.append(i)
        buy_y.append(short[999])
        trade_info.append([now_price,buy_amount])
    elif trade_info:
        if choice(short,long) == -1 and now_price / trade_info[0][0] >= 1 + r2 or now_price / trade_info[0][0] <= 1 - r1:
            jpy += now_price * trade_info[0][1]
            print("売却しました",i)
            sell_x.append(i)
            sell_y.append(short[999])
            del trade_info[0]
            
print("総資産:" + str(possetion(trade_info,now_price,jpy)))

'''
for data in X:
    now_price = data
    if jpy < 1000:
        buy_amount = jpy / now_price
    else:
        buy_amount = 1000 / now_price
    while i < len(trade_info):
        if now_price / trade_info[i][0] >= 1 + r2 or now_price / trade_info[i][0] <= 1 - r1:
            jpy += now_price * trade_info[i][1]
            del trade_info[i]
        i += 1
            
    if now_price / last_price < 1 - r3 and jpy > 0:
        jpy -= buy_amount * now_price
        trade_info.append([now_price,buy_amount])
    
    last_price = now_price
    if count >= 288 * 5:
        #last_price = hikaku[count - 120]
        last_price = max(hikaku[count-288*5:count])
    count += 1
    hikaku.append(data)
print(possetion(trade_info,now_price,jpy))
print(jpy)'''