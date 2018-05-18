import copy

from tqdm import tqdm


def describe(df):
    descr = df.describe()
    print(descr, "\n")
    return descr


def mean(df, attrs, printmean=False):
    stat = df.mean()
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printmean:
        i = 0
        for attr in attrs:
            print("Mean ", attr, ": ", ris[i])
            i += 1
    return ris


def max(df, attrs, printa=False):
    stat = df.max()
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("Max ", attr, ": ", ris[i])
            i += 1
    return ris


def min(df, attrs, printa=False):
    stat = df.min()
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("Min ", attr, ": ", ris[i])
            i += 1
    return ris


def std(df, attrs, printa=False):
    stat = df.std()
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("Stddev ", attr, ": ", ris[i])
            i += 1
    return ris


def q25(df, attrs, printa=False):
    stat = df.quantile(0.25)
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("25% ", attr, ": ", ris[i])
            i += 1
    return ris


def q50(df, attrs, printa=False):
    stat = df.quantile(0.5)
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("50% ", attr, ": ", ris[i])
            i += 1
    return ris


def q75(df, attrs, printa=False):
    stat = df.quantile(0.75)
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("75% ", attr, ": ", ris[i])
            i += 1
    return ris


def count(df, attrs, printa=False):
    stat = df.count()
    ris = []
    for attr in attrs:
        ris.append(stat[attr])
    if printa:
        i = 0
        for attr in attrs:
            print("Count ", attr, ": ", ris[i])
            i += 1
    return ris


def missing(df, attrs, printa=False):
    stat = len(df.index)-df.count()
    ris = dict()
    for attr in attrs:
        ris[attr] = stat[attr]
    if printa:
        print("MISSINGS: ")
        print(stat)
    return ris


def attributeslist(df):
    return list(df)


def misscorrelations(df, thr):
    attributes = attributeslist(df)
    missings = missing(df, attributes)
    datasnan = df.isna()
    length = len(df.index)
    for key in attributes:
        if missings[key] == 0:
            missings.__delitem__(key)
    keys = missings.keys()
    # when k1 misses, does k2 miss? ==> CONFIDENCE
    ris = dict()
    for k1 in keys:
        for k2 in keys:
            if k1 != k2:
                riskey = k1 + '>' + k2
                corr = 0
                for i in range(length):
                    if datasnan[k1][i] and datasnan[k2][i]:
                        corr += 1
                corr = corr / missings[k1]
                if corr >= thr:
                    ris[riskey] = corr
    return ris
