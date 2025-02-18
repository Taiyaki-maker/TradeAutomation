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
    
class Perceptron(object):
    """Perceptron classifier.

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
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

X_Z = zscore(X)
ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit(X_Z, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()

Xt_Z = zscore(Xt)

#prediction using sample data
y_pred = ppn.predict(Xt_Z)
#indicate missclassified samples
print('Missclassified samples:%d'%(yt != y_pred).sum())
print('Accuracy:'+str(1-(yt != y_pred).sum()/len(df2)))
