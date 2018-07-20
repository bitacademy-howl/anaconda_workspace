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

# 어떻게 동작할까???
# 0. (n + 1, n+1) 의 테이블 생성 (뭐 2차원 배열 등으로) 객체 생성
#     - 각 배열의 첫번째 (0, 1), (1, 0) 부터는 인덱스 채워넣음, (for문 or 객체)
# 1. column 추출 [첫번째, 두번쨰][첫번째, 세번째] 등의 (n by 2) permutaion
# 2. 각 경우의 수에 대한 correlation 수행
# 3. table[n, m] 에 매칭되는 컬럼명과 로우명 에 매칭되는 값들을 넣던지
#    아니면 포문돌때 배열이나 리스트등의 인덱스 접근 가능한 인덱스로 루프를 돌리면서 계산 즉시 넣음

# 대충 위처럼 하면 될텐데 구현 다 해야하는 것들...
# pandas 데이터프레임에 정의된 corr() 함수로 한번에 구할 수 있다.