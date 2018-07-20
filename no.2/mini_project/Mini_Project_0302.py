import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")
df = pd.read_csv('data_officesupplies.csv', header='infer', encoding='latin1')
print(df)

east_df = df[(df.get('Region') == 'East')]
west_df = df[df.get('Region') == 'West']
central_df = df[df.get('Region') == 'Central']

print(east_df)

plt.boxplot([west_df['Units'], east_df['Units'], central_df['Units']], 0)
plt.show()