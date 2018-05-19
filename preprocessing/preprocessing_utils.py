import pandas as pd
import dataset.dataset as d


def eliminate_zeros(df):
    for i in range(len(df)):
        if d.content_of(df, 'IsOpen', i) == 0:
            df.drop(df.index[i])
    return df
