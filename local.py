import dataset.utility as dsutil
import dataset.dataset as ds
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



f = ds.read_dataset("mean_var_pre_imputed.csv")


yrs = [dsutil.get_frame_in_range(f, 3, 2016, 2, 2017), dsutil.get_frame_in_range(f, 3, 2017, 2, 2018)]

regions = ds.values_of(f, 'Region')

data = {'2016/2017': {}, '2017/2018': {}}

plt_x = []


for year, (x, y, z, w) in enumerate([(3, 2016, 2, 2017), (3, 2017, 2, 2018)]):

    while (x - 1, y) != (z, w):

        plt_x.append("%s" % x)

        curr_m = dsutil.get_frame_in_range(yrs[year], x, y, x, y)

        key = '2016/2017' if year == 0 else '2017/2018'

        data[key][str(x)] = {}

        for r in regions:

            r_mean = (curr_m.loc[curr_m["Region"] == r])["NumberOfCustomers"].mean()
            print("m %s, y %s, region %s, mean %s" % (x, y, r, r_mean))

            data[key][str(x)][str(r)] = r_mean

        x = (x + 1 if x < 12 else 1)
        y = (y + 1 if x == 1 else y)



print(data)