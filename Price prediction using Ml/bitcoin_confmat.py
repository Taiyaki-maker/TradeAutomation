# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_csv("C:\sample\学習用データ.csv", header=None, encoding='utf-8')
df.tail()
df = df.iloc[::-1]
df.index = range(len(df))
X = np.array(df.iloc[0:7,2].values)
X = X.astype(float)
y = np.array([1 if df.iloc[7,2] > df.iloc[6,2] else -1])

df2 = pd.read_csv("C:\sample\訓練用データ.csv", header=None, encoding='utf-8')
df2.tail()
df2 = df2.iloc[::-1]
df2.index = range(len(df2))
Xt = np.array(df2.iloc[0:7,2].values)
Xt = Xt.astype(float)
yt = np.array([1 if df2.iloc[7,2] > df2.iloc[6,2] else -1])

count = 1
while count < len(df) - 7:
    x = df.iloc[count:count + 7,1].values
    x = x.astype(float)
    X = np.vstack((X,x))
    y = np.append(y,1 if df.iloc[count + 7,1] > df.iloc[count + 6,1] else -1)
    count += 1
    
count = 1
while count < len(df2) - 7:
    xt = df2.iloc[count:count + 7,2].values
    xt = xt.astype(float)
    Xt = np.vstack((Xt,xt))
    yt = np.append(yt,1 if df2.iloc[count + 7,2] > df2.iloc[count + 6,2] else -1)
    count += 1

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
X_Z = zscore(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=100, eta=0.01).fit(X_Z, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

'''ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X_Z, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')'''

# plt.savefig('images/02_11.png', dpi=300)
plt.show()

Xt_Z = zscore(Xt)

#テストデータで予測
y_pred = ada1.predict(Xt_Z)
confmat = confusion_matrix(y_true=yt, y_pred=y_pred)
print(confmat)
#誤分類のサンプルを表示
print('Missclassified samples:%d'%(yt != y_pred).sum())
print('Accuracy:'+str(1-(yt != y_pred).sum()/len(df2)))


