import joblib
import pandas as pd
import argparse
import os

from sklearn.metrics import accuracy_score

import model_dispatcher
import config

def run(fold,model):
    """
    fold: Number of fold wanted to train
    model: model wanted to train
    """
    #read the training data with folds
    df = pd.read_csv("input/mnist_train_folds.csv")
    #train_data is where kfold is different to actual fold number
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    #test_data is where kfold is equal to actual fold number
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    #Defining x_train,x_test,y_train,y_valid
    x_train = df_train.drop("label",axis=1).values
    y_train = df_train['label'].values

    x_valid = df_valid.drop('label',axis=1).values
    y_valid = df_valid['label'].values

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
