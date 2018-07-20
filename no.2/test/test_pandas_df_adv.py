import pandas as pd
import numpy as np
import os


os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")
df0 = pd.read_csv('data_iris.csv', header='infer',encoding = 'latin1')

df = df0.drop(columns = 'Species')

# 시발 쩐다...!!!
# 데이터 프레임이 엄청난 점....
# 코릴레이션을 통째로 다 구해서 표현할 수 있다.

# 확실히 분석가에게 엄청 좋을듯...!!!!!!!!
print(np.round(df.corr(),3))

