from math import log2
import pandas as pd
import numpy as np

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
