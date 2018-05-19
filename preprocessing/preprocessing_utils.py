from sklearn import preprocessing as pr
import dataset.dataset as d
import pandas as p
import numpy as np


def eliminate_IsOpen_zeros(df):
    df = df[(df[['IsOpen']] != 0).all(axis=1)]
    return df


def add_avg_per_shop(df):
    ids = d.values_of(df, 'StoreID')
    means = dict()
    for id in ids:
        try:
            _ = means[str(id)]
        except KeyError:
            means[str(id)] = average_per_shop(df, id)

    df['meanshop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'meanshop', means[str(d.content_of(df, 'StoreID', i))])
    return df


def average_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.mean()


def add_std_per_shop(df):
    ids = d.values_of(df, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = std_per_shop(df, id)

    df['mean_std_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'mean_std_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def std_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.std()


def add_max_per_shop(df):
    ids = d.values_of(df, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = max_per_shop(df, id)

    df['max_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'max_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def max_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.max()


def add_min_per_shop(df):
    ids = d.values_of(df, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = min_per_shop(df, id)

    df['min_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'min_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def min_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.min()





