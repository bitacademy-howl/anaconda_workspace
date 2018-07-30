import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from scipy.stats import itemfreq


os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_iris.csv', header='infer',encoding='latin1')

print(df.shape)


# 종 컬럼 분리
X=np.array(df.drop(columns='Species'))

Y=np.array(df.Species)

# 종별 카운팅 및 확인
table = itemfreq(Y)
plt.bar(table[:,0],table[:,1],color = 'blue')
plt.title('Category Frequency')
plt.show()

# 종별 각각의 변수들의 상관관계 확인
mycolors = {'setosa':'green', 'virginica':'blue', 'versicolor':'red'}
plots = pd.plotting.scatter_matrix(df,c=np.vectorize(mycolors.get)(Y),alpha=0.8,marker='o',s=10)

print(len(plots))
print(plots[0][0])
print(plots[0][1])
print(plots[0][2])
print(plots[0][3])

#
# for plot in plots:
#     plt.subplot(plot)
#     plt.show()