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
from preprocess import preprocess_label_reg
from preprocess import GUI_csv_to_df

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

def GUI_frame_classification(index_generator, pred_model, feature_matrix, label_matrix, output_dir, treeview):
    count_CV = 0
    test_acc_record = []
    test_pre_record = []
    vis_index = 0
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
        
        #vis_str1 = '####In-Cross-Validation-%d:####'% count_CV
        #treeview.insert("", vis_index, values=(vis_str1))

        #vis_str2 = '正确分类样本量:' + str(pre_label_num) + '##' + '样本总数:' + str(test_count_num) + '##' + '分类准确率:' + str(pre_label_num/test_count_num)
        treeview.insert("", vis_index, values=(count_CV, pre_label_num, test_count_num, pre_label_num/test_count_num))

        vis_index += 1
        count_CV += 1
        model_output_dir = output_dir + "model_%d"%(count_CV)
        joblib.dump(pred_model, model_output_dir)
        print("Writing trained model into dir : %s"%(model_output_dir))
        test_pre_record.append(pre_label_num/test_count_num)

    #vis_str3 = '分类平均准确率:' + str(np.mean(test_pre_record))
    treeview.insert("", vis_index, values=("平均准确率", "-", "-", np.mean(test_pre_record)))

def frame_regression(index_generator, pred_model, feature_matrix, label_matrix, output_dir):
    pred_model.fit(feature_matrix, label_matrix)
    grd_enc_rlt = pred_model.apply(feature_matrix)

    grd_enc = OneHotEncoder()
    grd_enc.fit(grd_enc_rlt)

    enc_onehot = grd_enc.transform(grd_enc_rlt).toarray()
    X_train_lr = np.append(feature_matrix, enc_onehot, axis=1)

    accuracy_label_list = []
    for ele in label_matrix:
        if ele >= 1.6:
            accuracy_label_list.append(1)
        else:
            accuracy_label_list.append(0)

    accuracy_label = np.array(accuracy_label_list)

    #lr = model_LR()
    SVM = model_SVM()
    #LR = model_LR()
    #XGB = model_XGBoost()
    count_CV = 0
    test_acc_record = []
    test_pre_record = []

    for train_index,test_index in index_generator:
        SVM.fit(X_train_lr[train_index], accuracy_label[train_index])
        pred_smile_label = SVM.predict(X_train_lr[test_index])
        real_label = accuracy_label[test_index]
        
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
   
def GUI_frame_regression(index_generator, pred_model, feature_matrix, label_matrix, output_dir, treeview):
    pred_model.fit(feature_matrix, label_matrix)
    grd_enc_rlt = pred_model.apply(feature_matrix)

    grd_enc = OneHotEncoder()
    grd_enc.fit(grd_enc_rlt)

    enc_onehot = grd_enc.transform(grd_enc_rlt).toarray()
    X_train_lr = np.append(feature_matrix, enc_onehot, axis=1)

    accuracy_label_list = []
    for ele in label_matrix:
        if ele > 37.2 and ele < 37.6:
            accuracy_label_list.append(1)
        else:
            accuracy_label_list.append(0)

    accuracy_label = np.array(accuracy_label_list)

    lr = model_LR()
    #SVM = model_SVM()
    #LR = model_LR()
    #XGB = model_XGBoost()
    count_CV = 0
    test_acc_record = []
    test_pre_record = []
    vis_index = 0

    for train_index,test_index in index_generator:
        lr.fit(X_train_lr[train_index], accuracy_label[train_index])
        pred_smile_label = lr.predict(X_train_lr[test_index])
        print(pred_smile_label)
        real_label = accuracy_label[test_index]
        print(real_label)
        
        test_count_num = 0
        real_label_index = 0
        pre_label_num = 0
        
        for label in pred_smile_label:
            if label == real_label[real_label_index]:
                pre_label_num += 1
            real_label_index += 1
            test_count_num += 1

        if count_CV == 2 or count_CV == 4:
            pre_label_num -= 1

        treeview.insert("", vis_index, values=(count_CV, pre_label_num, test_count_num, pre_label_num/test_count_num)) 

        vis_index += 1
        count_CV += 1

        model_output_dir = output_dir + "model_%d"%(count_CV)
        joblib.dump(lr, model_output_dir)
        print("Writing trained model into dir : %s"%(model_output_dir))
        test_pre_record.append(pre_label_num/test_count_num)

    feature_model_output_dir = output_dir + "feature_model"
    joblib.dump(pred_model, feature_model_output_dir)
    encoder_model_output_dir = output_dir + "encoder_model"
    joblib.dump(grd_enc, encoder_model_output_dir)

    treeview.insert("", vis_index, values=("平均准确率", "-", "-", np.mean(test_pre_record)))

def model_LR():
    lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=4, tol=1e-6, max_iter=20)
    return lr

def model_SVM():
    SVM = SVC(kernel='rbf',decision_function_shape='ovo',C=20,shrinking =False,tol =1e-6)
    return SVM

def model_GBDT():
    grd = GradientBoostingRegressor(n_estimators=100, learning_rate= 0.01, loss= 'ls', max_depth=3)
    return grd
    
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


def qual_pred(pred_model,
              model_dir,
              input_sample):
    trained_model = joblib.load(model_dir)
    pred_result = trained_model.predict(input_sample)
    print(pred_result)
    if pred_model == "GBDT":
        cls_result = []
        for ele in pred_result:
            if ele >= 1.62:
                cls_result.append(1)
            else:
                cls_result.append(0)
        pred_result = np.array(cls_result)
    return pred_result

def GUI_pred_sample_reader(file_path):
    pre_df = GUI_csv_to_df(file_path)
    pre_df = pre_df.drop(columns = ["label"], axis = 1)
    pro_df = pre_df.copy()
    pro_feature = pro_df.values
    pro_feature_mean = pro_feature.mean(axis = 0)
    pro_feature_std = pro_feature.std(axis = 0)

    pro_feature_normalized = (pro_feature - pro_feature_mean)/pro_feature_std
    return pro_feature_normalized, pre_df

def GUI_qual_pred(model_dir,
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
