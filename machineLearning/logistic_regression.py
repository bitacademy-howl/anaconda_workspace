# 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법

# 관측값은 True of False
# Y 가 1이 될 확률을 나타내는 S를 정의하고 : f(S)
# 그 S 는 선형회귀의 방법을 따른다.

# Confusion Matrix (CM)
#  ===> P(Predict | Actual) 을 true/false 표로 만든 2차 행렬

# 로지스틱 회귀의 정확도(Accuracy)

# A = Diagonal of CM / All of CM


# 로지스틱 회귀는 종속변수의 값이 2가지 상태일 경우에만 적용가능


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import itemfreq

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")
df = pd.read_csv('data_customer.csv')

# print(df.shape)
# print(df)

# print(df.columns)

X = np.array(df.drop(columns='PREGNANT'))
Y = np.array(df.get('PREGNANT'))

if X is not None and Y is not None:
    # print(X)
    # print(Y)

    # 아래는 Y 변수에 저장된 값들의 빈도를 matrix 로 변환

    table = itemfreq(Y)

    print(table.shape)
    print(table)

    # axis of X
    x_ticks = ['0','1']
    # 테이블의 2번째 컬럼 값들은 해당 값의 빈도수
    # plt.bar(x_ticks, table[:,1],color = 'blue')
    plt.bar(table[:, 0], table[:, 1], color='blue')

    plt.title('Category Frequency')
    plt.show()

    # 테스트용 변수와 학습용 변수의 나눔
    # 시드에 의한 랜덤넘버 생성이므로 인덱스의 값이 동일
    # 테스트 사이즈 30% of total number
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    # 트레이닝 데이터로부터 로지스틱회귀 수행
    glm = LogisticRegression()
    glm.fit(X_train, Y_train)
    print(X_train)

    Y_pred_train = glm.predict(X_train)
    Y_pred_test = glm.predict(X_test)

    conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test)
    print(conf_mat)

    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)
    sensitivity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    print('Accuracy    = ' + str(np.round(accuracy, 2)))
    print('Sensitvity  = ' + str(np.round(sensitivity, 2)))
    print('Specificity = ' + str(np.round(specificity, 2)))

    Y_pred_test_prob = glm.predict_proba(X_test)[:, 1]

    # threshold = 0.6
    threshold = 0.5
    Y_pred_test_val = (Y_pred_test_prob > threshold).astype(int)
    conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test_val)
    print(conf_mat)

    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)
    sensitivity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    print('Accuracy    = ' + str(np.round(accuracy, 2)))
    print('Sensitvity  = ' + str(np.round(sensitivity, 2)))
    print('Specificity = ' + str(np.round(specificity, 2)))

    t_grid = np.linspace(0.0, 1.0, 100)
    true_positive = []
    false_positive = []

    for threshold in t_grid:
        Y_pred_test_val = (Y_pred_test_prob > threshold).astype(int)
        conf_mat = metrics.confusion_matrix(Y_test, Y_pred_test_val)
        sensitivity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
        specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
        true_positive.append(sensitivity)
        false_positive.append(1 - specificity)

    plt.plot(false_positive, true_positive, c='red', linewidth=1.0)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve')
    plt.show()
else:
    print('X, Y 값이 입력되지 않음')