import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
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

table = itemfreq(Y)
print(table)

IPT = preprocessing.Imputer()
X = IPT.fit_transform(X)

X = preprocessing.scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)



C_grid = [0.001, 0.01, 0.1, 1, 10]
gamma_grid = [0.001, 0.01, 0.1, 1]
parameters = {'C': C_grid, 'gamma' : gamma_grid}
gridCV = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10);
gridCV.fit(X_train, Y_train)
best_C = gridCV.best_params_['C']
best_gamma = gridCV.best_params_['gamma']

print("SVM best C : " + str(best_C))
print("SVM best gamma : " + str(best_gamma))

SVM_best = SVC(C=best_C,gamma=best_gamma)
SVM_best.fit(X_train, Y_train);
Y_pred = SVM_best.predict(X_test)
print( "SVM best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))