import pandas as pd
import numpy as np
import os

df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
print(df)
print(df.shape)
print(df.info())
print(df.tail())

print(df.describe())
print((df.isnull()).sum(axis=0))

pd.plotting.scatter_matrix(df[['A', 'B', 'C', 'D']], alpha=0.2)

# pd.