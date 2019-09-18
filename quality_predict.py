import xlrd
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from preprocess import label_eng
from preprocess import csv_to_df
from preprocess import feature_eng

def cv_generator(feature_matrix, label_matrix):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    index_generator = kfold.split(feature_matrix, label_matrix)
    return index_generator

def frame_classification(index_generator, pred_model, feature_matrix, label_matrix, output_dir):
    count_CV = 0
    test_acc_record = []
    test_pre_record = []
    print(" *** Prediction model *** ")
    print( pred_model)

    for train_index,test_index in index_generator:

        pred_model.fit(feature_matrix[train_index], label_matrix[train_index])
        pred_smile_label = pred_model.predict(feature_matrix[test_index])
        real_label = label_matrix[test_index]
        
        test_count_num = 0
        real_label_index = 0
        pre_label_num = 0
        
        for label in pred_smile_label:
            if label == real_label[real_label_index]:
                pre_label_num += 1
            real_label_index += 1
            test_count_num += 1
        
        print('#### In Cross Validation %d: ####'% count_CV)
        count_CV += 1
        print('NumofIns Precisely Classified : ',pre_label_num,'\t',
              'NumofIns : ',test_count_num,'\t',
              'Pre_Accuracy : ',pre_label_num/test_count_num,'\t',)

        model_output_dir = output_dir + "model_%d"%(count_CV)
        joblib.dump(pred_model, model_output_dir)
        print("Writing trained model into dir : %s"%(model_output_dir))
        test_pre_record.append(pre_label_num/test_count_num)

    print('mean of NumofIns precisely classified',np.mean(test_pre_record))

def frame_regression(index_generator, pred_model, feature_matrix, label_matrix):
    pass
    
def model_LR():
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=4, tol=1e-6, max_iter=20)
    return lr

def model_SVM():
    SVM = SVC(kernel='rbf',decision_function_shape='ovo',C=20,shrinking =False,tol =1e-6)
    return SVM

def model_GBDT():
    pass
    
def model_XGBoost():
    XGB_C = XGBClassifier(
    #booster = 'gblinear',
    #objective='multi:softmax',
    #num_class=7,#必须要考虑到0的情况。这个数据集里面没有零
    n_estimators=200,
    max_depth=4,
    min_child_weight = 5,
    scale_pos_weight = 5,
    num_boost_round =5,
    max_delta_step=1000,
    alpha =2,
    eta=1
    #colsample_bytree=0.9
    #gamma=5,
    #process_type='update'
    )
    return XGB_C

def pred_sample_reader(sample_path,
                       sample_name,
                       pred_model,
                       feature_mean,
                       feature_std):
    pre_df = csv_to_df(sample_path, sample_name, pred_model)
    pro_df = pre_df.drop(columns = ["label"], axis = 1)
    pro_df['ApxRs'] = pro_df['Ap']*pro_df['Rs']
    pro_df['AexRs'] = pro_df['Ae']*pro_df['Rs']
    pro_df['AexAp'] = pro_df['Ae']*pro_df['Ap']
    pro_df['ApxRsxAe'] = pro_df['Ap']*pro_df['Rs']*pro_df['Ae']
    pro_feature = pro_df.values
    pro_feature_normalized = (pro_feature - feature_mean)/feature_std
    return pro_feature_normalized, pre_df


def qual_pred(model_dir,
              input_sample):
    trained_model = joblib.load(model_dir)
    pred_result = trained_model.predict(input_sample)
    return pred_result

def main():
    pre_df = csv_to_df("./dataset/", "data.csv")
    feature_matrix = feature_eng(pre_df)
    label_matrix = label_eng(pre_df)
    generator = cv_generator(feature_matrix, label_matrix)
    #running_model = model_LR()
    #running_model = model_SVM()
    running_model = model_XGBoost()
    frame_classification(generator, running_model, feature_matrix, label_matrix)

if __name__ == "__main__":
    main()
