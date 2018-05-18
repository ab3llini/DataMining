import sklearn
from sklearn.preprocessing.imputation import Imputer
import numpy as np
import dataset.dataset as d

def full_preprocess(ds):
    set_missings(ds, 'Events', 'none')
    set_missings(ds, 'CloudCover', 0)

    ds = impute_mean(ds, 'Max_VisibilityKm')
    ds = impute_mean(ds, 'Mean_VisibilityKm')
    ds = impute_mean(ds, 'Min_VisibilitykM')
    # Delete not imputed columns
    ds.__delitem__('Max_Gust_SpeedKm_h')
    return ds


def impute_mean(df, attr):
    imp = Imputer(missing_values="NaN", strategy="mean")
    imp.fit(df[[attr]])
    df[attr] = imp.transform(df[[attr]]).ravel()
    return df


def set_missings(ds, attr, val):
    datasnan = ds.isna()
    for i in range(len(ds)):
        if datasnan[attr][i]:
            ds.set_value(i, attr, val)
