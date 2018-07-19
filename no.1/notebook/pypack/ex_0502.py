import pandas as pd
import numpy as np
import os

os.chdir(r"D:\1. stark\0. 강의 게시판 자료\7. 분석, 시각화\data")
df = pd.read_csv('data_studentlist_en.csv', header='infer',encoding = 'latin1')

bloodType = df['bloodtype']
gender = df['gender']

print(bloodType.value_counts())