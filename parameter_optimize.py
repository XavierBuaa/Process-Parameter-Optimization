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
from quality_predict import model_LR
from quality_predict import model_SVM
from quality_predict import model_GBDT
from quality_predict import model_XGBoost

def para_optimize(running_model,
                  feature,
                  label,
                  feature_mean,
                  feature_std,
                  input_sample):
    running_model.fit(feature, label)
    sample_count = input_sample.shape[0]
    norm_sample = (input_sample - feature_mean)/feature_std
    quality_label = np.ones(sample_count)
    final_input_format = np.column_stack((norm_sample, quality_label))
    pred_result = running_model.predict(final_input_format)
    return pred_result

def op_sample_reader(sample_path,
                     sample_name,
                     col_name,
                     feature_mean,
                     feature_std):
    pre_df = csv_to_df(sample_path, sample_name)
    origin_col = pre_df[col_name]
    feature_df = pre_df.drop(columns = [col_name], axis = 1)
    pre_feature = feature_df.values

    raw_feature = pre_feature[:,0:-1]
    raw_feature_normalized = (raw_feature - feature_mean)/feature_std
    return raw_feature_normalized, pre_df, origin_col

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
    
