import sys
from argparse import ArgumentParser

import info_gain
import config_utils
import quality_predict
from preprocess import label_eng
from preprocess import csv_to_df
from preprocess import feature_eng
from preprocess import op_feature_ext
from parameter_optimize import para_optimize
from parameter_optimize import op_sample_reader


class Config(object):
    """
    Config object for hyper parameters
    """


config = Config()
config = config_utils.read_config_file("config.ini", "ProcessParameterOptimization", config)

arg_parser = ArgumentParser(description="training args in model")
arg_parser.add_argument("--run_infogain",
                        nargs="?", 
                        default=config.run_infogain,
                        help="running InfoGain or not")
arg_parser.add_argument("--run_qualitypredict",
                        nargs="?", 
                        default=config.run_qualitypredict,
                        help="running QualityPredict or not")
arg_parser.add_argument("--run_parameteroptimize",
                        nargs="?", 
                        default=config.run_parameteroptimize,
                        help="running ParameterOptimize or not")
arg_parser.add_argument("--file_path",
                        default=config.file_path,
                        help="input data file path")
arg_parser.add_argument("--file_name",
                        default=config.file_name,
                        help="input data file name")
arg_parser.add_argument("--pred_model",
                        default=config.pred_model,
                        help="running model of quality prediction")
arg_parser.add_argument("--output_dir",
                        default=config.output_dir,
                        help="output dir of trained model")
arg_parser.add_argument("--pred_file_name",
                        default=config.pred_file_name,
                        help="prediction file name")
arg_parser.add_argument("--opt_file_name",
                        default=config.opt_file_name,
                        help="optimization file name")
arg_parser.add_argument("--opt_parameter",
                        default=config.opt_parameter,
                        help="optimize parameter")
args = arg_parser.parse_args()

def main(args):
    print(" ****** Running function list ****** ")
    if args.run_infogain is True:
        print("Priority of parameters influencing product quality")
    if args.run_qualitypredict is True:
        print("Product quality predict")
    if args.run_parameteroptimize is True:
        print("Process parameter optimize")

    print(" ****** Loading trianing data from %s ******"%(args.file_path + args.file_name))
    raw_df = csv_to_df(args.file_path, args.file_name)
    print(" *** Head of original dataset *** ")
    print(raw_df.head())
    feature_matrix, ori_feature_mean, ori_feature_std = feature_eng(raw_df)
    label_matrix = label_eng(raw_df)

    if args.run_infogain is True:
        print(" ****** Priority of parameters influencing product quality ****** ")
        columns_entropy = [(col, info_gain.calcu_each_gain(raw_df[col], raw_df)) for col in raw_df.iloc[:, :-1]]
        print(" *** Information Gain of process parameter *** ")
        print(columns_entropy)    

    if args.run_qualitypredict is True:
        print(" ****** Product quality predict ****** ")
        generator = quality_predict.cv_generator(feature_matrix, label_matrix)
        running_model = quality_predict.model_XGBoost()
        print(" *** Training process *** ")
        quality_predict.frame_classification(generator, running_model, feature_matrix, label_matrix, args.output_dir)


        pred_norm_feature, pre_pred_df = quality_predict.pred_sample_reader(args.file_path,
                                                               args.pred_file_name,
                                                               ori_feature_mean,
                                                               ori_feature_std)
        pred_model_dir = args.output_dir + "model_1"
        pred_result = quality_predict.qual_pred(pred_model_dir, pred_norm_feature)

        print(" *** Product quality predict *** ")
        print("Origin process parameter")
        print(pre_pred_df)
        print("Prediction quality result")
        print(pred_result)

    if args.run_parameteroptimize is True:
        print(" ****** Process parameter optimize ****** ")
        feature, label, feature_mean, feature_std = op_feature_ext(raw_df, args.opt_parameter)
        running_model = quality_predict.model_XGBoost()
        input_sample, ori_fea, ori_col = op_sample_reader(args.file_path,
                                                          args.opt_file_name,
                                                          args.opt_parameter,
                                                          feature_mean,
                                                          feature_std)
        op_result = para_optimize(running_model,
                                  feature,
                                  label,
                                  input_sample)
        print(" *** Process parameter optimize *** ")
        print("Origin process parameter")
        print(ori_fea)
        print("Optimized %s"%(args.opt_parameter))
        print(op_result)



if __name__ == "__main__":
    args = arg_parser.parse_args()
    main(args)

