# -*- coding: utf-8 -*-
from fbprophet import Prophet # ライブラリの読み込み
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

x = []
count = 1
df = pd.read_csv("C:\sample\学習用データ.csv", header=None, encoding='utf-8')
df.tail()
X = np.array(df.iloc[1:1400,0].values)
#X = X.astype(float)
y = np.array(df.iloc[1:1400,1].values)
y = y.astype(float)
#X_Z = zscore(X)
#y_Z = zscore(y)
data = pd.DataFrame({'ds':X, 'y':y})

df2 = pd.read_csv("C:\sample\訓練用データ.csv", header=None, encoding='utf-8')
df2.tail()
Xt = np.array(df2.iloc[1:124,0].values)
yt = np.array(df2.iloc[1:124,2].values)
    
# 列名の変更
# インスタンス化
model = Prophet()
# 学習
model.fit(data)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)
test_fit = forecast.iloc[1673:1796,1].values
test_train = df2.iloc[1:124,2].values
test_fit = test_fit.astype(int)
output = pd.DataFrame({'予測値':test_fit, '実際の値':test_train})
for i in range(len(test_fit)):
    x.append(count)
    count += 1
plt.plot(x, test_fit, marker="o", color = "red", linestyle = "--")
plt.plot(x, test_train, marker="v", color = "blue", linestyle = ":");
