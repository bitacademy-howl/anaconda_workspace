import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics, preprocessing
from scipy.stats import itemfreq

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")
# df = pd.read_csv('data_spam.csv', header='infer',encoding='ISO-8859-1')
df = pd.read_csv('data_spam.csv', header='infer',encoding='latin1')

X=np.array(df.drop(columns=['email_id','is_spam']))
Y=np.array(df.is_spam)

print(Y.shape)

header = df.columns
headerX = df.drop(columns=['email_id','is_spam']).columns


# 아래 3행은 (true, false)로 정의되는 is_spam 이라는 컬럼값들을 0 과 1로 재지정하고, 빈도수를 측정한 2X2 테이블을 생성한다.
# 레이블을 0 ~ n-1 까지의 숫자로 표현해줌
LE = preprocessing.LabelEncoder()
# Fit label encoder and return encoded labels
Y = LE.fit_transform(Y)
table = itemfreq(Y)
print(table)

# 널값 제거 (imputer)

# strategy : 디폴트는 mean!!! - 없는 값을 해당 axis 의 mean 값으로 채워 넣음
# 세 가지 가능한 옵션
# The imputation strategy.
#
# If “mean”, then replace missing values using the mean along the axis.
# If “median”, then replace missing values using the median along the axis.
# If “most_frequent”, then replace missing using the most frequent value along the axis.

# axis : integer, optional (default=0)
# The axis along which to impute.
# If axis=0, then impute along columns.
# If axis=1, then impute along rows.

IPT = preprocessing.Imputer()
# 값을 X 테이블에 적용
X = IPT.fit_transform(X)
print('X before scaling : ', X)
X = preprocessing.scale(X)
print('X after scaling : ', X)

# 테스트 데이터 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 트리에 적용해볼 최종 트리의 depth 배열 (Node Level)
depth_grid = np.arange(2,31,2)
parameters = {'max_depth' : depth_grid}

# 적합한 모델을 찾기위해 하나하나 적용해보고 최적의 트리를 생성
# 여기서 최적이란 cv(Cross Validation : 교차 검증)을 통해 오류를 가장 최소화 하는 구조
gridCV = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
gridCV.fit(X_train, Y_train)
# 탐색이 완료 되면 gridCV 객체에 best_params_['max_depth'] 라는 멤버(딕셔너리)로 기록된다.
best_depth = gridCV.best_params_['max_depth']

print('최적 트리 노드레벨 : ', best_depth)
print("Tree best depth : " + str(best_depth)) # 같은 말

# 최적 depth 구한 후에 학습데이터 입력 후 학습
DTC_best = DecisionTreeClassifier(max_depth=best_depth)
DTC_best.fit(X_train, Y_train)

# 학습한 데이터로 테스트......대부분 학습모델객체.predict()
Y_pred = DTC_best.predict(X_test)

# 결과값의 정확성 출력....(실제 결과 대비 예측결과)
print( "Tree best accuracy : " + str(np.round(metrics.accuracy_score(Y_test, Y_pred),3)))

################################## 아래 에이다부스트 살펴볼 것!!! #####################################
# AdaBoost 적용 (교차검증 최적화 포함):
# 디폴트: base_estimator=DecisionTreeClassifier

estimator_grid = np.arange(1, 30, 5)
depth_grid = np.arange(1, 10, 2)
parameters = {'n_estimators': estimator_grid, 'max_depth': depth_grid}
gridCV = GridSearchCV(RandomForestClassifier(), param_grid=parameters, cv=10)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_depth = gridCV.best_params_['max_depth']

print("Random Forest best n estimator : " + str(best_n_estim))
print("Random Forest best depth : " + str(best_depth))

RF_best = RandomForestClassifier(max_depth=best_depth,n_estimators=best_n_estim,random_state=3)
RF_best.fit(X_train, Y_train)
Y_pred = RF_best.predict(X_test)
print( "Random Forest best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))

estimator_grid = np.arange(30, 80, 10)
learning_rate_grid = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
parameters = {'n_estimators': estimator_grid, 'learning_rate': learning_rate_grid}
gridCV = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, cv=10)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_learn_rate = gridCV.best_params_['learning_rate']

print("Ada Boost best n estimator : " + str(best_n_estim))
print("Ada Boost best learning rate : " + str(best_learn_rate))

AB_best = AdaBoostClassifier(n_estimators=best_n_estim,learning_rate=best_learn_rate,random_state=3)
AB_best.fit(X_train, Y_train);
Y_pred = AB_best.predict(X_test)
print( "Ada Boost best accuracy : " + str(np.round(metrics.accuracy_score(Y_test,Y_pred),3)))