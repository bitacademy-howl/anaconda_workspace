import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_male_physical_measurements.csv', header='infer',encoding='latin1')

df.shape

# df.head(5)
# header = df.columns

X=np.array(df)
kmeans = KMeans(n_clusters=3)
result = kmeans.fit(X).labels_
mycolor = []
for i in result:
    if i == 0:
        mycolor.append('red')
    elif i == 1:
        mycolor.append('green')
    else:
        mycolor.append('blue')

print(X[:,0])
print(X[:,1])
plt.scatter(X[:,0],X[:,1],marker="o",alpha=0.7, s=10,c=mycolor)
plt.xlabel("Body_Fat")
plt.ylabel("Density")
plt.show()

plt.scatter(X[:,2],X[0:,3],marker="o",alpha=0.7, s=10,c=mycolor)
plt.xlabel("X2")
plt.ylabel("X3")
plt.show()

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

