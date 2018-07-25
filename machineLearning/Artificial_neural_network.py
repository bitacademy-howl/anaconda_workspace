import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, preprocessing
from scipy.stats import itemfreq

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_spam.csv', header='infer',encoding='latin1')
X=np.array(df.drop(columns='is_spam'))
Y=np.array(df.is_spam)
header = df.columns
headerX = df.drop(columns='is_spam').columns

LE = preprocessing.LabelEncoder()
Y = LE.fit_transform(Y)

IPT = preprocessing.Imputer()
X = IPT.fit_transform(X)

X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

hidden_layer_config = [30, 30, 30]
alpha_grid = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
parameters = {'alpha': alpha_grid}

MLP = MLPClassifier(solver='lbfgs',hidden_layer_sizes=hidden_layer_config)
gridCV = GridSearchCV(MLP, parameters, cv=10)
gridCV.fit(X_train, Y_train)
best_alpha = gridCV.best_params_['alpha']

print("MLP best alpha : " + str(best_alpha))

MLP_best = MLPClassifier(solver='lbfgs',alpha=best_alpha,hidden_layer_sizes=hidden_layer_config)
MLP_best.fit(X_train, Y_train)
Y_pred = MLP_best.predict(X_test)
print( "MLP best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))