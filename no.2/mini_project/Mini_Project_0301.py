import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
PATH = r'D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data'
FILENAME = 'data_coffee.csv'

os.chdir(PATH)
df = pd.read_csv(FILENAME, header='infer',encoding = 'latin1')

df.isnull

print(df.info())
# sizeOfsite 결측치 제거
df2 = df.dropna(axis=0, subset=['sizeOfsite'])

print(df2)

print('결측치 제거 전 과 후의 shape 비교')
print('제거 전 : ', df.shape)
print('제거 후 : ', df2.shape)

plt.hist(df2.get('sizeOfsite'),bins=20,color='green', density=True)
plt.show()


result = df2.get('sizeOfsite')[df2.get('sizeOfsite') < 500]

plt.hist(result,bins=20,color='green', density=True)
plt.show()