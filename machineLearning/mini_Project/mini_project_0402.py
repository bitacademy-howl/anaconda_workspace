import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

path = r'D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data'
os.chdir(path)

df = pd.read_csv("data_kbo2015.csv")
print(df)

X = np.array([df['H']])
Y = np.array([df['HR']])

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
print(X.shape)
print(Y.shape)

print(X, Y, type(X))

lm = LinearRegression()
#
lm.fit(X, Y)
# #
print(lm.coef_)
#
print(lm.intercept_)
# #
R2 = lm.score(X,Y)
abs_corr = np.sqrt(R2)
#
# #
plt.scatter(X[:], Y[:], c = 'g', s = 15, alpha=0.5)
plt.xlabel("H")
plt.ylabel("HR")
plt.show()
