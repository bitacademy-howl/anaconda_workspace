import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

import os

# sklearn 에서 제공하는 학습용 데이터셋
data = load_boston()
print('=============================================================================')
print('================================ 데이터 타입 =================================')
print(type(data))
print('=============================================================================')

print('=============================================================================')
print(type(data.keys()), data.keys())
print('=============================================================================')

print('=============================== =설명서= ==================================')
print(data['DESCR'])
print('=============================================================================')

# 실제 값들만 존재하는 데이터셋
print('================================데이터 셋=====================================')
X = data['data']
print(X)
print('=============================================================================')

# 실제 데이터 필드에 컬럼명이 들어있지 않다.
print('=============================================================================')
header = data['feature_names']
print(header)

# 제공되는 데이터셋에 가격은 별도로 target 으로 제공되므로 dataframe을 만들때는 합쳐서 만든다....
print('=============================================================================')
Y = data['target']
Y = Y.reshape(-1, 1)
print(type(Y), Y)
print('=============================================================================')
# 실제 사용될 데이터 프레임 : 아직 헤더 포함되지 않음
df = pd.DataFrame(np.append(X, Y, axis=1))
print(df)
print('=============================================================================')


# 헤더에 header와 PRICE 컬럼명 추가
df.columns = np.append(header,'PRICE')

# 데이터 프레임에 헤더 추가
# 데이터프레임의 확인
print(df.head(5))
print(df.tail(5))


# 여러 통계치의 종합 선물세트
result_desc = df.describe()
print(result_desc)

#######################################################################################################
# 여기서 잠깐 번외로 통계치를 가지고
# 1. 박스플롯 그려보고
# 2. 분포도 그려보고

# # 1. 가격 분포도
# plt.hist(df['PRICE'],bins=100,color='green', density=True)
# plt.show()
# # 2.
# plt.boxplot([df['PRICE']],0)
# plt.show()

# 일단 이건 계속 해보고 생각해보쟈....
#######################################################################################################
# 각각의 컬럼간 상관관계
corr_df = np.round(df.corr(),3)
print(corr_df)

# ,marker='o',s=10
pd.plotting.scatter_matrix(df,alpha=0.8, diagonal='kde')

# os.chdir(r'D:\1. stark\temp')
#
# df.to_csv('data.csv',index=True)