import evaluation.evaluation as eva
import dataset.dataset as d
import numpy as np
import pandas as pd
import preprocessing.preprocessing_utils as preu

def gen_pandas_cols():
    shops_col = []
    months_col = []
    sales_col = []
    for i in range(len(shops)):
        for j in range(len(months)):
            shops_col.append(shops[i])
            months_col.append(months[j])
            sales_col.append(totp[i][j])
    return np.array(shops_col), np.array(months_col), np.array(sales_col)


name_precitions_csv = "final_sales_predictions.csv"
name_original_csv = "final_for_sales_test_r.csv"

preds = d.to_numpy(d.read_dataset(name_precitions_csv)).squeeze()


orig = d.read_dataset(name_original_csv)
dates = d.to_numpy(orig[['Date']]).squeeze()
regions = d.to_numpy(orig[['Region']]).squeeze()
ids = d.to_numpy(orig[['StoreID']]).squeeze()


error, totp, totr, shops, months = eva.region_error(preds, preds, regions, ids, dates, True)

print(error)
print(totp)
print(shops)
print(months)


sh, mo, sa = gen_pandas_cols()
final_sub = pd.DataFrame()
final_sub['StoreID'] = pd.Series(sh)
final_sub['Month'] = pd.Series(mo)
final_sub['NumberOfSales'] = pd.Series(sa)

print(final_sub)


# ADDING MORE STATISTICS
trainset = "final_for_sales_train.csv"

trainds = d.read_dataset(trainset)
final_sub_stats = preu.mean_sales_per_month_per_shop(final_sub, trainds)
final_sub_stats['Ratio'] = final_sub_stats['NumberOfSales']/final_sub_stats['MeanSalesPerShopPerMonth']
print(final_sub_stats)
d.save_dataset(final_sub_stats, "final_sub_with_stats.csv")

