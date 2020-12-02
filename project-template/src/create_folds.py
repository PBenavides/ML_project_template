#This file will create a new file in the input/ called mnist_train_folds.csv
#With the difference that the csv is shuffled and has a new column called kfold.

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def read_train_data():
    """
    Function to read and concat training data that is split across multiples csv.
    """
    os.chdir("input")
    extension = 'csv'
    train_filenames = [i for i in glob.glob('*.{}'.format(extension)) if "train" in i]
    concat_df = pd.concat([pd.read_csv(file) for file in train_filenames])
    return concat_df

def create_folds(data, n_splits=5):
    """
    To split a dataset where every class is balanced.
    """
    data["kfold"] = -1
    #We randomized our data.
    data = data.sample(frac=1).reset_index(drop=True)
    #Get the number of bins by Sturges rule
    num_bins = np.floor(1 + np.log2(len(data))) 
    #Get the bins intervals labels, not the extension <n;n+k)
    data.loc[:, "bins"] = pd.cut(
        data["label"], bins=int(num_bins), labels=False
    )
    #Then we'll get a balance Folds with StratifiedKFolds, by the bins previously generated
    kf = StratifiedKFold(n_splits=n_splits)
    #we will replace the data kfold column, to indentify which row belong which test_index
    for fold, (train_index, test_index) in enumerate(kf.split(X=data, y=data['bins'].values)):
        data.loc[test_index,'kfold'] = fold
    data = data.drop("bins", axis=1)
    return data

if __name__ == "__main__":
    data = read_train_data()
    kfolded_data = create_folds(data=data, n_splits = 5)
    #then we will write on input folder
    kfolded_data.to_csv('mnist_train_folds.csv',index=False)
    print("Kfolds Done!")