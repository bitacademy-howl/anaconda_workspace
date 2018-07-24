import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, preprocessing
from scipy.stats import itemfreq
from sklearn.decomposition import PCA


os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")
df = pd.read_csv('data_spam.csv', header='infer',encoding='latin1')
df.shape
df.head(5)

X=np.array(df.drop(columns=['is_spam', 'email_id']))
Y=np.array(df.is_spam)

header = df.columns
headerX = df.drop(columns=['is_spam', 'email_id']).columns

table = itemfreq(Y)
print(table)
# table = np.unique(Y, return_counts=True)
# print(table)

plt.bar(table[:,0],table[:,1],color = 'blue')
plt.title('Category Frequency')
plt.show()
print(table)

LE = preprocessing.LabelEncoder()
Y = LE.fit_transform(Y)

table = itemfreq(Y)
print(table)


IPT = preprocessing.Imputer()
X = IPT.fit_transform(X)

np.round(df.describe(),5)

# None 이 있으면 채워넣음?
# 표준화 ===> 변수의 단위차에 의한 표현의 단위를 적절히 조정 <------------------>   =>   <--->
X = preprocessing.scale(X)

np.round(pd.DataFrame(X,columns=headerX).describe(),3)

# 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# k = 5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, Y_train)
Y_pred = knn5.predict(X_test)

print(metrics.confusion_matrix(Y_test,Y_pred))
print("------------------------")
print(metrics.accuracy_score(Y_test,Y_pred))
print( "Accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

# k = 100
knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train, Y_train)
Y_pred = knn100.predict(X_test)
print(metrics.confusion_matrix(Y_test,Y_pred))
print("------------------------")
print(metrics.accuracy_score(Y_test,Y_pred))
print( "Accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

accs = []
k_grid = np.arange(1,51,1)
for k in k_grid:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accs.append(metrics.accuracy_score(Y_test,Y_pred))

print(k_grid)

# 데이터로 k = 3 이 최적합 k 임을 알수 있다.
plt.scatter(k_grid, accs, c='red',marker='o',s=10,alpha=0.7)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k')
plt.show()

k_grid = np.arange(1,51,1)

# 거리에 따른
weights = ['uniform','distance']
parameters = {'n_neighbors':k_grid, 'weights':weights}

########################################################################################################################
# 여기부터 볼 것!!!
########################################################################################################################
gridCV = GridSearchCV(KNeighborsClassifier(), parameters, cv = 10)                                      # cv : 교차 검증
gridCV.fit(X_train, Y_train)

best_k = gridCV.best_params_['n_neighbors']
best_w = gridCV.best_params_['weights']

print("Best k : " + str(best_k))
print("Best weight : " + best_w)

knn_best = KNeighborsClassifier(n_neighbors=best_k, weights = best_w)
knn_best.fit(X_train, Y_train)
Y_pred = knn_best.predict(X_test)
print( "Best Accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

# k-NN + PCA 적용
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=3)
knn_best = KNeighborsClassifier(n_neighbors=best_k, weights = best_w)
knn_best.fit(X_train, Y_train)
Y_pred = knn_best.predict(X_test)
print( "Best Accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))