import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_activity_training.csv', header='infer',encoding='latin1')


# 컬럼명만 따로 뽑아...
# 1st : 헤더와 데이터 분리
# 2nd : 헤더 [] 리스트에서 *time* 이 포함된 인자 제거
# 3rd : np.array 에서 df.drop(columns=[]) 넣어주면 됨.....

data = np.array(df)
headers = df.columns

remove_col = []
for header in headers:
    if header.find('time') != -1:
        remove_col.append(header)
    else:
        pass

remove_col.extend(["Unnamed: 0", "user_name", "new_window"])
# print(remove_col)

dataframes = df.drop(columns=remove_col)


# 결측도 계산
df_none = dataframes.isnull().mean(axis=0)
# 위는 dataframes.isnull() 함수가 가져오는 것은 True or False
# Boolean 값을 mean 수치계산 할 때 숫자로 변경되어 1 과 0의 빈도수와 동일해지는 원리를 이용해서 결측도를
# 계산한 것!!


print(df.count())

# df_none = dataframes.isnull().sum(axis=0)
# df_none2 = df_none/dataframes.shape[0]
# 이것도 같은 결과.. - -----> Dataframe / 상수 ===>>>> 모든 cell 에 반복적으로 연산결과 수행
#                                                      numpy.ndarray 객체와 같이 동작,..,.,ㅣ.,.,.

# null_dataf = np.array(pd.isnull(df).mean(axis=0) < 0.97)
# result = df.loc[:, null_dataf]

for row in dataframes:
    pass



# headerX = df.drop(columns=remove_col).columns
#             ---------------------------
#                       ^
#                       df
#            -----------------------------------
#                              ^
#                             col



# 결측치가 존재하는 행 버리기....






#   ..................................................................................................  #


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



# 1. 결측치 집계
#
# 2. 정상값이 3% 미만인 컬럼 제거
#
# 3. time 이 들어간 컬럼 제거
#
# 4. unnamed:0, user_name, new_window 컬럼 제거
#
# 5. KNN을 적용하여 train & test 한다.... 정확도 계산까지
#
# 6. knn 매개변수 최적화 후 다시 학습


# 데이터 내용 (참고)
# classe
# 해당 (row)인간이 취하는 activity 종류

# X=np.array(df.drop(columns=''))
# Y=np.array(df.is_spam)
# header = df.columns
# headerX = df.drop(columns='is_spam').columns
