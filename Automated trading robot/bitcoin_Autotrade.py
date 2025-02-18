# -*- coding: utf-8 -*-
import requests
import json
from zaifapi.impl import ZaifTradeApi
import numpy as np
import time

currency = "eth_jpy"

#ビットコインの日本円価格を取得
url = "https://api.zaif.jp/api/1/last_price/" + currency
response = requests.get(url)

#取引情報の定義
key = "ここにapiキーを入力"
secret = "ここにシークレットキーを入力"
zaif = ZaifTradeApi(key, secret)

r1 = 0.05
r2 = 0.01
#直前の価格と比較する場合は0.005に設定する
r3 = 0.005

i = 0
count = 0
trade_info = []
last_price = json.loads(response.text)['last_price']
comparism = []

def possetion(trade_info,now_price,jpy):
    sum = 0
    for i in range(len(trade_info)):
        sum += trade_info[i][1] * now_price
    return sum + jpy

while True:
    jpy = zaif.get_info2()['funds']['jpy']
    btc = zaif.get_info2()['funds']['ETH']
    now_price = json.loads(response.text)['last_price']
    print("総資産" + str(jpy + btc * now_price))
    print("現在の価格:" + str(now_price))
    if jpy < 1000:
        buy_amount = jpy / now_price
    else:
        buy_amount = 1000 / now_price

    while i < len(trade_info):
        if now_price / trade_info[i][0] >= 1 + r2 or now_price / trade_info[i][0] <= 1 - r1:
            zaif.trade(currency_pair=currency, action="ask", price=now_price, amount=buy_amount)
            print('売り')
            del trade_info[i]
        i += 1           
    if now_price / last_price < 1 - r3 and jpy > 0:
        zaif.trade(currency_pair=currency, action="bid", price=now_price, amount=buy_amount)
        print('買い')
        trade_info.append([now_price,buy_amount])
        
    last_price = now_price
    '''if count >= 120:
        last_price = comparism[count - 120]'''
    comparism.append(now_price)
    count += 1
    time.sleep(60)