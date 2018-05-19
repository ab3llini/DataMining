import dataset.dataset as d
import numpy as np


def tss(df, attr):
    arr = d.to_numpy(df[[attr]]).squeeze()
    mean = arr.mean()
    return np.sum(np.square(arr-mean))


def rss(df, preds, attr):
    arr = d.to_numpy(df[[attr]]).squeeze()
    return np.sum(np.square(arr-preds))


def r2(df, preds, attr):
    return 1 - rss(df, preds, attr) / tss(df, attr)
