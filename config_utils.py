import configparser

class Config(object):
    """
    Config object for hyper parameters
    """

def read_config_file(filename, section_name, config_obj):
    """
    Construct config dict from config.ini
    """
    data = {}
    config = configparser.ConfigParser()
    try:
        with open(filename, 'r') as confile:
            config.readfp(confile)
            for sect in config.sections():
                for (key, value) in config.items(sect):
                    data[key] = value
    except Exception:
        print("read config file fail", Exception)
    return reconfig(config_obj, data, section_name)

def reconfig(config_obj, config_dict, section_name):
    """
    Construct config object from config dict classified by section_name
    """
    if section_name == "ProcessParameterOptimization":
        if config_dict["run_infogain"] == "True":
            config_obj.run_infogain = True
        else:
            config_obj.run_infogain = False 

        if config_dict["run_qualitypredict"] == "True":
            config_obj.run_qualitypredict = True
        else:
            config_obj.run_qualitypredict = False 

        if config_dict["run_parameteroptimize"] == "True":
            config_obj.run_parameteroptimize = True
        else:
            config_obj.run_parameteroptimize = False 

        config_obj.file_path = str(config_dict["file_path"])
        config_obj.file_name = str(config_dict["file_name"])
        config_obj.pred_model = str(config_dict["pred_model"])
        config_obj.output_dir = str(config_dict["output_dir"])
        config_obj.pred_file_name = str(config_dict["pred_file_name"])

        config_obj.opt_file_name = str(config_dict["opt_file_name"])
        config_obj.opt_parameter = str(config_dict["opt_parameter"])

    return config_obj

def main():
    config = Config()
    config = read_config_file("config.ini", "ProcessParameterOptimization", config)
    print(config.run_infogain)
    print(config.run_qualitypredict)
    print(config.run_parameteroptimize)
    print(config.file_path)
    print(config.file_name)
    print(config.pred_model)
    print(config.opt_file_name)
    print(config.opt_parameter)

if __name__ == "__main__":
    main()
