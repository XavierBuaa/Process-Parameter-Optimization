import numpy as np
import pandas as pd

from math import log2
from preprocess import csv_to_df

def get_counts(data):
    total = len(data)
    results = {}
    for d in data:
        results[d] = results.get(d, 0) + 1
    return results, total

# 计算信息熵
def calcu_entropy(data):
    results, total = get_counts(data)
    ent = sum([-1.0*v/total*log2(v/total) for v in results.values()])
    return ent

# 计算每个feature的信息增益
def calcu_each_gain(column, update_data):
    total = len(column)
    grouped = update_data.iloc[:, -1].groupby(column)
    temp = sum([len(g[1])/total*calcu_entropy(g[1]) for g in list(grouped)])
    return calcu_entropy(update_data.iloc[:, -1]) - temp

def main():
    raw_df = csv_to_df("./dataset/", "data.csv")
    columns_entropy = [(col, calcu_each_gain(raw_df[col], raw_df)) for col in raw_df.iloc[:, :-1]]
    print(columns_entropy)    

if __name__ == "__main__":
    main()
