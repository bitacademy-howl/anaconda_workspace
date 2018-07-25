import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_KOSPI200_en.csv', header='infer',encoding='latin1')
# print(df.shape)

# X = 수익률, 리스크
# conpanies = COMPANY


X=np.array(df.iloc[:,[3,4]])
# print(X, X.shape)
#
# print(X[:, 0])

# print(df.iloc[:, [3]], type(df.iloc[:, [3]]))
# plt.ylabel('Sin')

plt.scatter(X[:,0], X[:,1], c='red',alpha=0.5)
plt.show()

companies = np.array(df.iloc[:,1])

# 비지도 학습이기때문에 학습데이터와 test 데이터를 나누지 않는다...(??????????????????)
# kmeans = KMeans(n_clusters=2)
kmeans = KMeans()
clusters = kmeans.fit(X).labels_
centers = kmeans.cluster_centers_
table = np.unique(clusters,return_counts=True)

# print('Cluster Sizes :')
# print(table)
# print('Cluster centers :')
# print(centers)

# for i in range(len(centers)):
#     print("Companies in group {} :".format(i))
#     print("-----------------------------------")
#     print(companies[clusters==i])
#     print ("\n")


# 의미 : 중심점을 정해 (모든 중심점에 대한 각각의)거리의 제곱의 합
#        ==> 원소의 갯수만큼의 클러스터로 나눌경우 TTSW 는 0
#        여기서는 clusters = 198

def total_ss_within(X, centers, clusters):
    # "Total Sum of Squares Within"을 계산하여 최적화된 클러스터 갯수를 알아낸다.
    N_clusters = centers.shape[0]
    # print(N_clusters)
    N_columns = centers.shape[1]
    N_rows = X.shape[0]
    ref_centers = np.zeros((N_rows, N_columns))
    for n in range(N_clusters):
        indices = (clusters == n)
        for j in range(N_columns):
            ref_centers[indices,j] = centers[n,j]
    return np.sum((X-ref_centers)**2.0)

# Kmeans 최근종가,수익률, 리스크, 베타 기준으로 클러스터 갯수의 최적화:
# 구분짓기 위한 변수 : (최근 종가, 수익률, 리스크, 베타)

X=np.array(df.iloc[:,[2,3,4,5]])

n_cluster = np.array(range(2,20))
# n_cluster = np.array(range(2,198))
total_ssw = np.array([])

for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    # print(kmeans)
    clusters = kmeans.fit(X).labels_
    # print(kmeans.fit(X))
    print(clusters)
    centers = kmeans.cluster_centers_

    total_ssw = np.append(total_ssw, total_ss_within(X,centers,clusters))

plt.plot(n_cluster,total_ssw,color='blue',marker='o',linestyle='dashed',linewidth=1,markersize=5)
plt.show()




plt.scatter(X[:, 0], X[:, 1], c='red',alpha=0.5)
plt.show()

companies = np.array(df.iloc[:,1])

# 비지도 학습이기때문에 학습데이터와 test 데이터를 나누지 않는다...(??????????????????)
kmeans = KMeans(n_clusters=5)
# kmeans = KMeans()
clusters = kmeans.fit(X).labels_
centers = kmeans.cluster_centers_
table = np.unique(clusters,return_counts=True)

# 만약 위의 예제에서 수익율과 리스크로만 구성된
# def total_ss_within(X, centers, clusters):
#     # "Total Sum of Squares Within"을 계산하여 최적화된 클러스터 갯수를 알아낸다.
#     N_clusters = centers.shape[0]
#     N_columns = centers.shape[1]
#     N_rows = X.shape[0]
#     ref_centers = np.zeros((N_rows, N_columns))
#     for n in range(N_clusters):
#         indices = (clusters == n)
#         for j in range(N_columns):
#             ref_centers[indices,j] = centers[n,j]
#     return np.sum((X-ref_centers)**2.0)
#
# X=np.array(df.iloc[:,[3,4]])
# n_cluster = np.array(range(2,40))
# total_ssw = np.array([])
# for n in n_cluster:
#     kmeans = KMeans(n_clusters=n)
#     clusters = kmeans.fit(X).labels_
#     centers = kmeans.cluster_centers_
#     total_ssw = np.append(total_ssw, total_ss_within(X,centers,clusters))
#
# plt.plot(n_cluster,total_ssw,color='blue',marker='o',linestyle='dashed',linewidth=1,markersize=5)
# plt.show()


# BDSCAN : 군집화에 필요한 parameters(엡실론, minpts) 중 엡실론은 pixcel 간 거리라고 표현하고 (distance = 1, minpts = 2)
#          라고 정의할 때, 모니터에 출력된 결과 영상의 군집화가 가능할 것!
#          단 각 군집을 이루는 요소의 값 (color) 의 threshold 비교가 이루어진 이후에 그 각 요소별 군집을 파악하는데 사용하여야 할 것!

#     minpts = 2 라는 것은 같은 색분포로 이어져 있는 상태라는 것!
#     distance = 1 이라는 것은 바로 옆의 픽셀을 비교하겠다는 것!
# 매우 strict 한 군집화에 속함....실제 이미지는 이런식으로 군집화 하면 제대로 이뤄지지 않을것!!
# (threshold 값을 넉넉하게 주지 않는다면,,,,,,,,) - ex) 음영, 색밀도(원근에 따른 - vectorized)

# 일단 색밀도로 분포도를 표현하고, 그것의 형태등을 군집화????
# ㄴㄴ...

# 색을 하나의 컬럼으로 주고,

# 군집화 한 후에 그것의 합을 형상비교????

# 아몰랑


