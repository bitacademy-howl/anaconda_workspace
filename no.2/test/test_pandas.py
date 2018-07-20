import pandas as pd
import os

path = r'D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data'
filename = 'data_studentlist_en.csv'

os.chdir(path)
df = pd.read_csv(filename, header='infer',encoding = 'latin1')

# 데이터 프레임 기초
print('################################## 타입 #########################################')
print(type(df))
print('################################## df.info #########################################')
print(df.info())
print('################################## df.head #######################################')
print(df.head(3))

print('################################## df.tail #######################################')
print(df.tail(3))

print('################################## df.column #######################################')
print(df.name)

# df.columns 는 이미 데이터 프레임에 정의된 속성이다....
# column 들을 별도로 관리하는 속성
# 때문에 아래와 같이 df.[속성] 값을 가져올 때, 컬럼명으로 columns 를 지정하였다면 가져올 수 없다.
# df['columns'] 로는 가능

# 이와같은 속성들이 더 있을것이므로
# 컬럼을 가져올 때, 속성 값에 직접 접근 (object.a 의 형태) 보다는 key값으로 접근하여 사용하도록 하자...!!
# 혹은 getter 의 사용

df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'columns', 'e']


print(df)
print('###################################################################################')
print(df.a)
# 아래는 df 객체의 속성을 불러온다.
print(df.columns)
print('###################################################################################')
print('1. df[columns]')
print(df['columns'])
print("2. df.get('columns')")
print(df.get('columns'))

print('###################################################################################')

