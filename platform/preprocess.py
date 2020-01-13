import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def preprocess_label(label):
    if label > 37.2 and label < 37.6:
        return 1
    else:
        return 0

def csv_to_df_raw(file_path):
    engine = create_engine(file_path)
    sql = 'select * from moment'
    raw_df = pd.read_sql_query(sql, engine)
    raw_df.rename(columns = {0:"moment", 1:"flatness", 2:"label"}, inplace = True)
    return raw_df

def csv_to_df(file_path):
    engine = create_engine(file_path)
    sql = 'select * from moment'
    raw_df = pd.read_sql_query(sql, engine)
    raw_df.rename(columns = {0:"moment", 1:"flatness", 2:"label"}, inplace = True)
    raw_df["label"] = raw_df.apply(lambda x : preprocess_label(x.label), axis = 1)
    return raw_df

def main():
    pre_df = csv_to_df()
    print(pre_df)

if __name__ == "__main__":
    main()
