## Utility functions to run training from the shell.

import joblib
import pandas as pd

import argparse
import os
import yaml
import logging 

from sklearn.metrics import accuracy_score
from data import preprocessing_data

import model_dispatcher
import config

#In order to use yaml confg file, we'll make this functions.

#def load_cfg(yaml_filepath):
#    """
#    yaml_filepath: string
#    
#    -----
#    Returns cfg: dict
#    """
#    with open(yaml_filepath, 'r') as stream:
#        cfg = yaml.load(stream)
#
#    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
#    return cfg
#
#def make_paths_absolute(dir_, cfg):
#    """
#    Make all values for keys ending with `_path` absolute to dir_.
#    """
#    for key in cfg.keys():
#        if key.endswith("_path"):
#            cfg[key] = os.path.join(dir_, cfg[key])
#            cfg[key] = os.path.abspath(cfg[key])
#            if not os.path.isfile(cfg[key]):
#                logging.error("%s does not exist.", cfg[key])
#        if type(cfg[key]) is dict:
#            cfg[key] = make_paths_absolute(dir_, cfg[key])
#    return cfg

def run(fold,model):
    """
    fold: int Number of fold wanted to train
    model: str model wanted to train
    """
    #read the training data with folds
    df = pd.read_csv(f"input/{config.PROJECT_NAME}_folds.csv")

    #train_data is where kfold is different to actual fold number
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    #test_data is where kfold is equal to actual fold number
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    #Defining x_train,x_test,y_train,y_valid
    x_train = df_train.drop(config.TARGET_NAME,axis=1).values
    y_train = df_train[config.TARGET_NAME].values

    x_valid = df_valid.drop(config.TARGET_NAME,axis=1).values
    y_valid = df_valid[config.TARGET_NAME].values

    #----------------------------------TRAINING----------------------------------------------
    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = accuracy_score(y_valid, preds)

    print(f"Fold={fold}, Accuracy={accuracy}")

    #saving the model
    joblib.dump(clf,
    os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__=="__main__":
    #we will specify arguments to run from a shell scripting.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(fold = args.fold,
        model = args.model)
