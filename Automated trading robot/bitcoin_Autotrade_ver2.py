# -*- coding: utf-8 -*-
import requests
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sympy
import time
import ccxt

# APIキー＆シークレットキー
apiKey  = "ここにAPIキーを入力"
secret  = "ここにシークレットキーを入力"

# bitbank取引所とAPI呼出
bitbank = ccxt.bitbank({'apiKey':apiKey,'secret':secret})

r1 = 0.01
r2 = 0.004

i = 0
count = 0
trade_info = [167,155]
comparism = []
chart_sec = 300
buy_x = []
buy_y = []
sell_x = []
sell_y = []

# CryptowatchでBTCの価格データを取得
def get_price(min, before=0, after=0):
	price = []
	params = {"periods" : min }
	if before != 0:
		params["before"] = before
	if after != 0:
		params["after"] = after
	response = requests.get("https://api.cryptowat.ch/markets/bitflyer/ethjpy/ohlc",params)
	data = response.json()
	
	if data["result"][str(min)] is not None:
		for i in data["result"][str(min)]:
			price.append({ "close_time" : i[0],
				"close_time_dt" : datetime.fromtimestamp(i[0]).strftime('%Y/%m/%d %H:%M'),
				"open_price" : i[1],
				"high_price" : i[2],
				"low_price" : i[3],
				"close_price": i[4] })
	return price

def balance():
    result = bitbank.fetch_balance()
    return result

def ticker(symbol):
    result = bitbank.fetch_ticker(symbol=symbol)
    return result
        
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
    #bitcoinの場合は80,300
    ε2 = 40
    ε3 = 150
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
    if (abs(short[999] - long[999]) < ε2 or flag or a1 > grad or a1 < -grad or dict[x] - 999 < ε and dict[x] >= 1000) and abs(short[999] - long[999]) < ε3:
        if short[999] > long[999] and a1 < 0:
            return -1
        elif short[999] < long[999] and a1 > 0:
            return 1
    return 0

def buy(bitbank,now_price):
    jpy = balance()['JPY']['total']
    for i in range(3):
        if jpy > 100:
            order = bitbank.create_order(
                            symbol = 'ETH/JPY',    #通貨
                            type   = 'market',      #注文方法：market(成行)、limit(指値)
                            side   = 'buy',        #購入(buy) or 売却(sell)
                            amount = (jpy / now_price) * 0.7,       #購入数量[BTC]
                            )
            time.sleep(0.5)
            jpy = balance()['JPY']['total']
            
def sell(bitbank):
    eth = balance()['ETH']['total']
    if eth >= 0.0001:
        order = bitbank.create_order(
                        symbol = 'ETH/JPY',    #通貨
                        type   = 'market',      #注文方法：market(成行)、limit(指値)
                        side   = 'sell',        #購入(buy) or 売却(sell)
                        amount = eth,       #購入数量[BTC]
                        )

#メッセージを引数として渡し、その内容を通知する関数send_line_notifyを定義
def send_line_notify(notification_message):
    line_notify_token = 'line notify api keys'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'{notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)


try:
    while True:
        # 価格チャートを取得
        price = get_price(chart_sec)
        Y = [i['close_price'] for i in price]
        df = pd.DataFrame(Y)
        short = df[0].ewm(span=48, adjust=False).mean()
        long = df[0].ewm(span=288, adjust=False).mean()
        now_price = ticker(symbol='ETH/JPY')['last']
        '''
        x = np.arange(0, 1000, 1)
        plt.plot(x, short, 'r-')
        plt.plot(x, long, 'b-')
        plt.plot(buy_x,buy_y,"o", markersize = 8, color = 'g')
        plt.plot(sell_x,sell_y,"o", markersize = 8, color = 'y')
        plt.show()'''
        if choice(short,long) == 1:
            buy(bitbank,now_price)
            trade_info.append(now_price)   
            buy_x.append(i)
            buy_y.append(short[999])
            send_line_notify(str(now_price) + "円でethを購入しました")
        elif trade_info:
            if choice(short,long) == -1 and now_price / trade_info[0] >= 1 + r2 or now_price / trade_info[0] <= 1 - r1:
                sell(bitbank)
                sell_x.append(i)
                sell_y.append(short[999])
                send_line_notify(str(now_price) + "円でethを売却しました")
                del trade_info[0]
        time.sleep(300)
except Exception as e:
    print(send_line_notify("エラーが発生しました、実行を停止します。エラー内容:" + str(e)))
