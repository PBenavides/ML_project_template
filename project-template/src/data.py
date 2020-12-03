## Functions to Load, Preprocessing, and Handling Data.
## All the functions here need to be aproved by and documented by notebooks Delivered-*.ipynb

import pandas as pd
import os, glob

def read_train_data():
    """
    Function to read and concat training data that is split across multiples csv.
    """
    os.chdir("input")
    extension = 'csv'
    train_filenames = [i for i in glob.glob('*.{}'.format(extension)) if "train" in i]
    concat_df = pd.concat([pd.read_csv(file) for file in train_filenames])
    return concat_df

def preprocessing_data(df):
    """
    A simple preprocessing function.
    """
    #Dealing with Nans
    df['Edad'].fillna(int(df['Edad'].mean()), inplace=True)
    df['P_embarque'].fillna('S',inplace=True)
    df['P_embarque'].fillna(df['P_embarque'].mode(), inplace=True)
    df['Tarifa'].fillna(df['Tarifa'].mean(),inplace=True)
    df.drop(['IdPasajero','Name','Ticket','Cabin'],axis=1, inplace=True)
    #Dealing with outliers
    outliers_to_repl = df[df['Tarifa'] > df['Tarifa'].quantile(.95)].index
    df.loc[outliers_to_repl, 'Tarifa'] = df['Tarifa'].quantile(.95)

    #Feature Engineering
    df['Miembros_de_fam'] = df['Hermanos'] + df['Padres_hijos'] + 1
    df['Viaja_solo'] = 1 
    df['Viaja_solo'].loc[df['Miembros_de_fam'] > 1] = 0

    df['es_niño'] = 0
    df.loc[(df['Edad'])<15,'es_niño'] = 1

    #Labeling & Categorization
    cat_to_nums = {"P_embarque":  {"S": 2, "C": 1, "Q":0},
               "Genero": {"male":0,"niño":1,"female":2}}

    df.replace(cat_to_nums, inplace=True)

    #ReScaling data
    return df