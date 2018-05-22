import sklearn
from sklearn.preprocessing.imputation import Imputer
import numpy as np
import dataset.dataset as d
import pandas as p

def full_preprocess(ds):
    """Exploits the full preprocessing of the original dataset.
    Categorical attributes are represented into one_hot encoding, missing values are handles and imputed and
    the column Max_Gust_SpeedKm_h is deleted.
    Returns a DataFrame object"""
    set_missings(ds, 'Events', 'none')
    ds = one_hot(ds, 'Events', split=True, header='Events_')
    ds = one_hot(ds, 'StoreType', header='StoreType_')
    ds = one_hot(ds, 'AssortmentType', header='AssortmentType_')
    set_missings(ds, 'CloudCover', 0)

    ds = impute_mean(ds, 'Max_VisibilityKm')
    ds = impute_mean(ds, 'Mean_VisibilityKm')
    ds = impute_mean(ds, 'Min_VisibilitykM')
    # Delete not imputed columns
    ds.__delitem__('Max_Gust_SpeedKm_h')
    ds.__delitem__('Events')
    ds.__delitem__('StoreType')
    ds.__delitem__('AssortmentType')
    return ds


def impute_mean(df, attr):
    """Imputes the given attribute of the given DataFrame with the mean strategy.
    Returns a DataFrame object"""
    imp = Imputer(missing_values="NaN", strategy="mean")
    imp.fit(df[[attr]])
    df[attr] = imp.transform(df[[attr]]).ravel()
    return df


def set_missings(ds, attr, val):
    """Sets the missing values of the given attribute of the given DataFrame with the given value val."""
    datasnan = ds.isna()
    for i in range(len(ds)):
        if datasnan[attr][i]:
            ds.set_value(i, attr, val)


def one_hot(ds, attr, header, split=False):
    """Transforms the given attribute of the given DataFrame object into one hot encoding.
    If you plan to use this, don't use split attribute.
    Returns a DataFrame object."""
    vals = d.values_of(ds, attr)
    if split:
        new_cols = []
        for v in vals:
            split = v.split("-")
            for s in split:
                if not new_cols.__contains__(s):
                    new_cols.append(s)
    else:
        new_cols = vals
    for new in new_cols:
        ds[header + new] = p.Series(np.zeros(len(ds)), ds.index)
        for i in range(len(ds)):
            if d.content_of(ds, attr, i).find(new) != -1:
                ds.set_value(i, header + new, 1)
    return ds


if __name__ == '__main__':
    ds = d.read_test_dataset()
    ds = full_preprocess(ds)
    d.save_dataset(ds, "imputed_test_ds_one_hot.csv")
