import pandas
import os.path
import numpy as np


def read_dataset():
    """Returns the original dataset"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return pandas.read_csv(os.path.join(_dir, "train.csv"))


def read_imputed_onehot_dataset():
    """Returns the imputed dataset with categorical attributes transformed into one-hot encoding"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return pandas.read_csv(os.path.join(_dir, "imputed_ds_one_hot.csv"))


def save_dataset(ds, name):
    """Saved the given dataset in THIS FOLDER with the given name."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    ds.to_csv(os.path.join(_dir, name), index=False)


def numeric_only(ds):
    """Returns a DataFrame object containing only the numerical columns of the given DataFrame"""
    return ds.select_dtypes(include=[np.number])


def nominal_only(ds):
    """Returns a DataFrame object containing only the categorical columns of the given DataFrame"""
    return ds.select_dtypes(exclude=[np.number])


def content_of(df, attr, row):
    """returns the content of the cell (row, attr) of the given DataFrame object"""
    return df.at[row, attr]


def values_of(df, attr):
    """returns the list of DISTINCT values of the given attribute of the given DataFrame"""
    ris = []
    for i in range(len(df)):
        if not ris.__contains__(content_of(df, attr, i)):
            ris.append(content_of(df, attr, i))
    return ris


if __name__ == '__main__':
    print(read_dataset().head(15))
    print(read_imputed_onehot_dataset().head(15))




