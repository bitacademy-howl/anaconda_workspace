import os

# 이거 import 또 하고 코드를 더 만드는 것보다....
# numpy 어차피 임모트 통째로 했으니 numpy 에 구현된 모든 기능을 그냥 쓰는게 낫다...
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
#
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def ShowMe(X):
    Y= 1.0 - X
    plt.imshow(Y, cmap='gray')
    plt.show()

os.chdir(r'D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data')

df = pd.read_csv("data_mnist_train_100.csv")
# df = pd.read_csv("data_mnist_train_100.csv")

# 요기까지 같이 구현하고 아래부터 동일하도록...
# 군데 구냥 문법 익힐 겸 구현해본다 연습...
########################################################################################################
# 리셰이프...ㅜㅠ
# for i in range (100)
# x1 = np.array([X[0]]).reshape(28, 28)
########################################################################################################
# 시벌 안쓰고 리스트로 해본다...
# change dimension 함수의 구현....
def ch_dimension(img):
    n = sqrt(len(img))
    # print(n, type(n), n == 28)
    result = []
    inner_result = []
    for x in img:
        if len(inner_result) == n:
            result.append(inner_result)
            inner_result = []
        inner_result.append(x)
    result.append(inner_result)
    return result


# data size 확인 후에 sqrt()

# 여기 포문 돌리면 됨

# nparr = np.array(df)

# 데이터 정렬
# 데이터가 순서 없이 들어가 있으므로 탐색을 용이하게 위해 sort...
# 여기서 정렬 기준 컬럼이 되는 '5' 라는 컬럼은 뒤에 (28X28 = 784) 배열에 담긴 픽셀정보가 표현하는
# 숫자를 의미하는 컬럼이다...
# 컬럼명은 걍 바꿀 수 있지만 아직 걍 놔둬본다.
df = df.sort_values(by='5')

# 데이터를 나눔 (ndarray 자료형은 인덱스로 접근이 가능하다....)
# 정렬한 데이터를 다시 새로운 어레이로 나눈 것이므로 이전의 sort 된 인덱스와는 별개로
# 새로운 인덱스를 가지며, 각 인덱스는 X : Y 매핑 된다.
img = df.drop(columns='5')
X = np.array(img)
Y = np.array(df['5'])

# 데이터의 확인....
# print(df)
# result = np.array(ch_dimension(X[3]))
# result = np.array(ch_dimension(X[0], sqrt(len(X[0]))))
# result = np.array(ch_dimension(nparr[0, 1:]))
# ShowMe(result)

# 이제 X 와 Y를 가지고 학습한다...
# 많은 알고리즘을 사용할 수 있겠지만, 회귀적 문제라고는 보기 어려우므로
# 일단 분류 알고리즘을 적용하도록 한다.

# 아래는 Random Forest 를 적용

# 테스트 데이터 나누기
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 위의 학습데이터 생성과 같은 방법으로 테스트 데이터 생성
df_test = pd.read_csv("data_mnist_test_100.csv")
df_test = df_test.sort_values(by='5')
test_img = df_test.drop(columns='5')
X_test = np.array(test_img)
Y_test = np.array(df_test['5'])

# 트리에 적용해볼 최종 트리의 depth 배열 (Node Level)
depth_grid = np.arange(2,100,2)
parameters = {'max_depth' : depth_grid}

# 적합한 모델을 찾기위해 하나하나 적용해보고 최적의 트리를 생성
# 여기서 최적이란 cv(Cross Validation : 교차 검증)을 통해 오류를 가장 최소화 하는 구조
gridCV = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
gridCV.fit(X, Y)
# # 탐색이 완료 되면 gridCV 객체에 best_params_['max_depth'] 라는 멤버(딕셔너리)로 기록된다.
# best_depth = gridCV.best_params_['max_depth']
#
# print('최적 트리 노드레벨 : ', best_depth)

# # 최적 depth 구한 후에 학습데이터 입력 후 학습
# DTC_best = DecisionTreeClassifier(max_depth=best_depth)
# DTC_best.fit(X, Y)
#
# # 학습한 데이터로 테스트......대부분 학습모델객체.predict()
# Y_pred = DTC_best.predict(X_test)
#
# # 결과값의 정확성 출력....(실제 결과 대비 예측결과)
# print("Tree best accuracy : " + str(np.round(metrics.accuracy_score(Y_test, Y_pred),3)))
#
# # 1.데이터를읽어온다: “data_mnist_train_100.csv” 등.
# # 3.머신러닝방법을적용해서training과testing을하고정확도를계산해본다.
# # 2018/7/17 ~ 2018/7/30
# # 4.비교적낮은정확도를향상시킬수있는방법을적용해본다.
# # 힌트: rotation 방법으로학습데이터의분량을키운다.
# # scipy.ndimage.interpolation의rotate함수사용.