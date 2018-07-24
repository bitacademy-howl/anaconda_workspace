import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
import os

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_number_nine.csv', header='infer',encoding='latin1')

print(df.shape)

df.head(5)


def ShowMe(X):
    Y= 1.0 - X
    plt.imshow(Y, cmap='gray')
    plt.show()

X = np.array(df)
ShowMe(X)
print(X)

def reducedPCA(X,nPC):
    pca = PCA(n_components = nPC)
    X_pca = pca.fit_transform(X)
    print(X_pca)
    return pca.inverse_transform(X_pca)

for nPC in [23, 10, 5, 3, 1]:
    Z = reducedPCA(X,nPC)
    # print(Z)
    print( "N# of PCs = " + str(nPC))
    ShowMe(Z)
# print(Z)

