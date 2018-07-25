import os



# 이거 import 또 하고 코드를 더 만드는 것보다....
# numpy 어차피 임모트 통째로 했으니 numpy 에 구현된 모든 기능을 그냥 쓰는게 낫다...
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
#
import numpy as np

def ShowMe(X):
    Y= 1.0 - X
    plt.imshow(Y, cmap='gray')
    plt.show()

os.chdir(r'D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data')

df = pd.read_csv("data_mnist_train_100.csv")


img = df.drop(columns='5')
X = np.array(img)
Y = np.array(df['5'])


# 요기까지 같이 구현하고 아래부터 동일하도록...
# 군데 구냥 문법 익힐 겸 구현해본다 연습...
########################################################################################################
# 리셰이프...ㅜㅠ
# for i in range (100)
# x1 = np.array([X[0]]).reshape(28, 28)
########################################################################################################
# 시벌 안쓰고 리스트로 해본다...
# change dimension 함수의 구현....
def ch_dimension(img, n):
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

result = np.array(ch_dimension(X[0], sqrt(len(X[0]))))
ShowMe(result)




# print(list(X[0]), '\n', len(list(X[0])))
# print(X[0], len(X[0]))

# print('ㅇㅇㅇㅇㅇㅇ: ',result1.shape)
#
# print(result1)
#



# print(df, type(df))
# ShowMe(data[0])

# print(df)



