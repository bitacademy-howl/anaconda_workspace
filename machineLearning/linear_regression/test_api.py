import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import itemfreq

X = np.array(['1번', '2번', '3번', '4번', '5번','6번','7번','8번','9번','10번'])
Y = np.array([1,2,3,4,5,6,7,8,9,10])

# 테스트용 변수와 학습용 변수의 나눔 (쌍으로 동일하게 추출 = because by seed)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

# print(X_train)
# print(Y_train)

X_train, X_test = train_test_split(X, test_size=0.25, random_state=2)
Y_train, Y_test = train_test_split(Y, test_size=0.25, random_state=2)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)
