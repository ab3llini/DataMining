import dataset.dataset as d
import preprocessing.preprocessing_utils as pre_u
import pandas as p
import preprocessing.imputation as imp
import numpy as np
import dataset.utility as utils

def build_sales_predictor_dataset(name):
    ds = d.read_imputed_onehot_dataset()
    ds = __prepare_sales_ds(ds)
    d.save_dataset(ds, name)


def build_cust_predictor_dataset(name):
    ds = d.read_imputed_onehot_dataset()
    ds = __prepare_customers_ds(ds)
    d.save_dataset(ds, name)


def __prepare_sales_ds(ds):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = pre_u.mean_std_sales_per_shop_per_day(ds)
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.add_avg_per_shop(ds)
    ds = pre_u.add_std_per_shop(ds)
    ds = pre_u.add_max_per_shop(ds)
    ds = pre_u.add_min_per_shop(ds)
    return ds


def __prepare_customers_ds(ds):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = pre_u.mean_std_cust_per_shop_per_day(ds)
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.add_avg_cust_per_shop(ds)
    ds = pre_u.add_std_cust_per_shop(ds)
    ds = pre_u.add_max_cust_per_shop(ds)
    ds = pre_u.add_min_cust_per_shop(ds)
    ds['NearestCompetitor'] = p.Series(np.array([1/ds['NearestCompetitor'][i] for i in ds.index.tolist()]), ds.index)
    return ds


def select_features(name, featlist, fname):
    ds = d.read_dataset(name)
    ds = ds[featlist]
    d.save_dataset(ds, fname)


if __name__ == '__main__':
    # build_sales_predictor_dataset("fully_preprocessed_ds.csv")
    #select_features("fully_preprocessed_ds.csv", ['IsHoliday', 'NearestCompetitor',
    #                                              'NumberOfSales', 'NumberOfCustomers', 'HasPromotions',
    #                                              'Date', 'MeanSalesPerShopPerDay',
    #                                              'StdSalesPerShopPerDay',
    #                                              'meanshop',
    #                                              'mean_std_shop',
    #                                              'max_shop','min_shop'], "fpd_select.csv")
    ds = d.read_dataset("fully_preprocessed_ds.csv")
    ds = utils.get_frame_in_range(ds, 1, 2018, 2, 2018)
    cust = d.read_dataset("customer_pred_jan_feb_LR_DEG1.csv")
    cust = d.to_numpy(cust[['0']]).squeeze()
    cust = np.array(cust, dtype=np.int32)
    cust[cust < 0] = 0
    ds['NumberOfCustomers'] = p.Series(cust, ds.index)
    d.save_dataset(ds, "fpd_with_customers.csv")
