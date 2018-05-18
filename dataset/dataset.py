import pandas
import os.path
import numpy as np

def read_dataset():
    _dir = os.path.dirname(os.path.abspath(__file__))
    return pandas.read_csv(os.path.join(_dir, "train.csv"))


def numeric_only(ds):
    return ds.select_dtypes(include=[np.number])


def nominal_only(ds):
    return ds.select_dtypes(exclude=[np.number])


def content_of(df, attr, row):
    return df.at[row, attr]


def values_of(df, attr):
    nominal_attrs = list(nominal_only(df))
    if not nominal_attrs.__contains__(attr):
        return []
    ris = []
    for i in range(len(df)):
        if not ris.__contains__(content_of(df, attr, i)):
            ris.append(content_of(df, attr, i))
    return ris


if __name__ == '__main__':
    print(read_dataset())




