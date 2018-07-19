import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\1. stark\0. 강의 게시판 자료\7. 분석, 시각화\data")

df = pd.read_csv('data_studentlist_en.csv', header='infer',encoding = 'latin1')

df.shape

df.head(5)

print(df)

print(df.loc[:,'height'].mean())

print(df['height'].mean())
print(df['height'].quantile(0.1))