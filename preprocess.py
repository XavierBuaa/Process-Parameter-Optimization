import os
import pandas as pd
import numpy as np

def csv_to_df(file_path, file_name, pred_model):
    file_dir = os.path.join(file_path, file_name)
    raw_df = pd.read_csv(file_dir, header = None)
    raw_df.rename(columns = {0:"Ae", 1:"Ap", 2:"Rs", 3:"Fz", 4:"label"}, inplace = True)
    raw_df.drop(columns = ['Fz'], axis = 1, inplace = True)
    if pred_model == "XGBoost":
        raw_df["label"] = raw_df.apply(lambda x : preprocess_label(x.label), axis = 1)
    else:
        raw_df["label"] = raw_df.apply(lambda x : preprocess_label_reg(x.label), axis = 1)
    return raw_df

#def csv_to_matrix(file_path, file_name):
#    pre_df = csv_to_df('./dataset/', 'data.csv')
#    matrix = pre_df.as_matrix()
#    return matrix

def preprocess_label(label):
    if label > 2:
        return 1
    else:
        return 0

def preprocess_label_reg(label):
    if label > 1.6:
        return 1
    else:
        return 0

def feature_eng(raw_df):
    pro_df = raw_df.drop(columns = ["label"], axis = 1)
    pro_df['ApxRs'] = pro_df['Ap']*pro_df['Rs']
    pro_df['AexRs'] = pro_df['Ae']*pro_df['Rs']
    pro_df['AexAp'] = pro_df['Ae']*pro_df['Ap']
    pro_df['ApxRsxAe'] = pro_df['Ap']*pro_df['Rs']*pro_df['Ae']
    pro_feature = pro_df.values
    pro_feature_mean = pro_feature.mean(axis = 0)
    pro_feature_std = pro_feature.std(axis = 0)
    pro_feature_normalized = (pro_feature - pro_feature_mean)/pro_feature_std
    return pro_feature_normalized, pro_feature_mean, pro_feature_std

def label_eng(raw_df):
    raw_label = raw_df['label'].values
    return raw_label

def op_feature_ext(raw_df, col_name):
    label_df = raw_df[col_name]
    label_feature = label_df.values
    feature_df = raw_df.drop(columns = [col_name], axis = 1)
    pre_feature = feature_df.values

    raw_feature = pre_feature[:,0:-1]
    quality_label = pre_feature[:,-1]

    raw_feature_mean = raw_feature.mean(axis = 0)
    raw_feature_std = raw_feature.std(axis = 0)
    raw_feature_normalized = (raw_feature - raw_feature_mean)/raw_feature_std
    raw_mix = np.column_stack((raw_feature_normalized, quality_label))

    return raw_mix, label_feature, raw_feature_mean, raw_feature_std

def main():
    pre_df = csv_to_df("./dataset/", "data.csv")
    #pre_matrix = csv_to_matrix('./dataset/', 'data.csv')
    feature_matrix = feature_eng(pre_df)
    label_matrix = label_eng(pre_df)
    print(type(feature_matrix))
    print(type(label_matrix))

if __name__ == "__main__":
    main()
