import xlrd
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from preprocess import csv_to_df
from preprocess import op_feature_ext
from preprocess import GUI_csv_to_df
from quality_predict import model_LR
from quality_predict import model_SVM
from quality_predict import model_GBDT
from quality_predict import model_XGBoost

def para_optimize(running_model,
                  feature,
                  label,
                  pred_model,
                  input_sample):
    running_model.fit(feature, label)
    sample_count = input_sample.shape[0]
    if pred_model == "XGBoost":
        quality_label = np.ones(sample_count)
    else:
        quality_label = np.ones(sample_count)
        for i in range(sample_count):
            quality_label[i] = 2.5
    final_input_format = np.column_stack((input_sample, quality_label))
    pred_result = running_model.predict(final_input_format)
    return pred_result

def op_sample_reader(sample_path,
                     sample_name,
                     col_name,
                     pred_model,
                     feature_mean,
                     feature_std):
    pre_df = csv_to_df(sample_path, sample_name, pred_model)
    origin_col = pre_df[col_name]
    feature_df = pre_df.drop(columns = [col_name], axis = 1)
    pre_feature = feature_df.values

    raw_feature = pre_feature[:,0:-1]
    raw_feature_normalized = (raw_feature - feature_mean)/feature_std
    return raw_feature_normalized, pre_df, origin_col

def GUI_op_training_data(file_name):
    raw_df = GUI_csv_to_df(file_name)
    
    label_df = raw_df["moment"]
    label_feature = label_df.values
    feature_df = raw_df.drop(columns = ["moment"], axis = 1)                          
    pre_feature = feature_df.values                                                   
    
    raw_feature = pre_feature[:,0:-1]                                                 
    quality_label = pre_feature[:,-1]                                                 
    
    raw_feature_mean = raw_feature.mean(axis = 0)                                     
    raw_feature_std = raw_feature.std(axis = 0)
    raw_feature_normalized = (raw_feature - raw_feature_mean)/raw_feature_std         
    raw_mix = np.column_stack((raw_feature_normalized, quality_label))
    return raw_mix, raw_feature_mean, raw_feature_std, label_feature

def GUI_op_sample_reader(file_path, feature_mean, feature_std):
    pre_df = GUI_csv_to_df(file_path)
    feature_df = pre_df.drop(columns = ["moment"], axis = 1)
    pre_feature = feature_df.values

    raw_feature = pre_feature[:,0:-1]
    raw_feature_normalized = (raw_feature - feature_mean)/feature_std
    return raw_feature_normalized, feature_df

def GUI_para_optimize(feature,
                      label,
                      input_sample):
    running_model = model_GBDT()
    running_model.fit(feature, label)
    sample_count = input_sample.shape[0]
    quality_label = np.ones(sample_count)
    for i in range(sample_count):
        quality_label[i] = 37.4
    final_input_format = np.column_stack((input_sample, quality_label))
    pred_result = running_model.predict(final_input_format)
    return pred_result

def main():
    pre_df = csv_to_df("./dataset/", "data.csv")
    feature, label, feature_mean, feature_std = op_feature_ext(pre_df, "Rs")
    running_model = model_XGBoost()
    input_sample, feature_df, origin_col = op_sample_reader("./dataset/",
                                                            "opt_data.csv",
                                                            "Rs",
                                                            feature_mean,
                                                            feature_std)
    op_result = para_optimize(running_model,
                              feature,
                              label,
                              feature_mean,
                              feature_std,
                              input_sample)
    print(input_sample)
    print(op_result)

    
if __name__  == "__main__":
    main()
    
