import os
import pandas as pd
import numpy as np
import pyfpgrowth

os.chdir(r"D:\1. stark\anaconda_workspace\no.2\머신러닝 알고리즘과 응용\data")

df = pd.read_csv('data_association.csv', header='infer',encoding='latin1')

print(df.shape)

print(df)

transactions = {}


for id, item in zip(df.customer_id, df.item):
    if id in transactions.keys():
        transactions[id] = transactions[id] + [item]
    else:
        transactions[id] = [item]

print(transactions)

trans_list=list(transactions.values())
print(trans_list)

patterns = pyfpgrowth.find_frequent_patterns(trans_list, 0.001)
print(patterns)

rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

print(rules)