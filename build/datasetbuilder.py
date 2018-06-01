import dataset.dataset as d
import preprocessing.preprocessing_utils as pre_u
import pandas as p
import preprocessing.imputation as imp
import numpy as np
import dataset.utility as utils

def build_sales_predictor_train_dataset(name):
    ds = d.read_imputed_onehot_dataset()
    ds = __prepare_sales_train_ds(ds)
    d.save_dataset(ds, name)


def build_cust_predictor_train_dataset(name):
    ds = d.read_imputed_onehot_dataset()
    ds = __prepare_customers_train_ds(ds)
    d.save_dataset(ds, name)


def build_sales_predictor_test_dataset(name):
    ds_tr = d.read_dataset("final_for_sales_train.csv")
    ds = d.read_imputed_onehot_test_dataset()
    ds = __prepare_sales_test_ds(ds, ds_tr)
    d.save_dataset(ds, name)


def build_cust_predictor_test_dataset(name):
    ds_tr = d.read_dataset("final_for_customer_train.csv")
    ds = d.read_imputed_onehot_test_dataset()
    ds = __prepare_customers_test_ds(ds, ds_tr)
    d.save_dataset(ds, name)


def __prepare_sales_train_ds(ds):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds['Date'] = ds['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ds['Month'] = ds['Date']
    ds['Month'] = ds['Month'].apply(lambda x: x.split("-")[1])
    ds = imp.one_hot_numeric(ds, 'Month', 'Month_')
    ds = imp.one_hot_numeric(ds, 'Region', 'Region_')
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.mean_std_sales_per_shop_per_day(ds)
    ds = pre_u.add_avg_per_shop(ds)
    ds = pre_u.add_std_per_shop(ds)
    ds = pre_u.add_max_per_shop(ds)
    ds = pre_u.add_min_per_shop(ds)
    ds = pre_u.mean_sales_per_month_per_region(ds)
    return ds


def __prepare_customers_train_ds(ds):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds['Date'] = ds['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ds['Month'] = ds['Date']
    ds['Month'] = ds['Month'].apply(lambda x: x.split("-")[1])
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = imp.one_hot_numeric(ds, 'Month', 'Month_')
    ds = imp.one_hot_numeric(ds, 'Region', 'Region_')
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.mean_std_cust_per_shop_per_day(ds)
    ds = pre_u.add_avg_cust_per_shop(ds)
    ds = pre_u.add_std_cust_per_shop(ds)
    ds = pre_u.add_max_cust_per_shop(ds)
    ds = pre_u.add_min_cust_per_shop(ds)
    ds = pre_u.mean_cust_per_month_per_shop(ds)
    ds = pre_u.mean_cust_per_month_per_region(ds)
    return ds


def __prepare_sales_test_ds(ds, dfrom):
    ds['NumberOfSales'] = p.Series(np.zeros(len(ds)), ds.index)
    ds['NumberOfCustomers'] = p.Series(np.zeros(len(ds)), ds.index)
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds['Date'] = ds['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ds['Month'] = ds['Date']
    ds['Month'] = ds['Month'].apply(lambda x: x.split("-")[1])
    ds = imp.one_hot_numeric(ds, 'Month', 'Month_')
    ds = imp.one_hot_numeric(ds, 'Region', 'Region_')
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.mean_std_sales_per_shop_per_day(ds, dfrom)
    ds = pre_u.add_avg_per_shop(ds, dfrom)
    ds = pre_u.add_std_per_shop(ds, dfrom)
    ds = pre_u.add_max_per_shop(ds, dfrom)
    ds = pre_u.add_min_per_shop(ds, dfrom)
    ds = pre_u.mean_sales_per_month_per_region(ds, dfrom)
    return ds


def __prepare_customers_test_ds(ds, dfrom):
    ds['NumberOfSales'] = p.Series(np.zeros(len(ds)), ds.index)
    ds['NumberOfCustomers'] = p.Series(np.zeros(len(ds)), ds.index)
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds['Date'] = ds['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ds['Month'] = ds['Date']
    ds['Month'] = ds['Month'].apply(lambda x: x.split("-")[1])
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = imp.one_hot_numeric(ds, 'Month', 'Month_')
    ds = imp.one_hot_numeric(ds, 'Region', 'Region_')
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.mean_std_cust_per_shop_per_day(ds, dfrom)
    ds = pre_u.add_avg_cust_per_shop(ds, dfrom)
    ds = pre_u.add_std_cust_per_shop(ds, dfrom)
    ds = pre_u.add_max_cust_per_shop(ds, dfrom)
    ds = pre_u.add_min_cust_per_shop(ds, dfrom)
    ds = pre_u.mean_cust_per_month_per_shop(ds, dfrom)
    ds = pre_u.mean_cust_per_month_per_region(ds, dfrom)
    return ds


def select_features(name, featlist, fname):
    ds = d.read_dataset(name)
    ds = ds[featlist]
    d.save_dataset(ds, fname)


def create_all_finals():
    build_sales_predictor_train_dataset("final_for_sales_train.csv")
    build_cust_predictor_train_dataset("final_for_customer_train.csv")
    build_sales_predictor_test_dataset("final_for_sales_test.csv")
    build_cust_predictor_test_dataset("final_for_customer_test.csv")


if __name__ == '__main__':
     # create_all_finals()
     attrs_sal_tr = list(d.read_dataset("final_for_sales_train.csv"))
     attrs_cus_tr = list(d.read_dataset("final_for_customer_train.csv"))
     pre_u.reorder_datas_cols("final_for_sales_test.csv", attrs_sal_tr, "final_for_sales_test_r.csv")
     pre_u.reorder_datas_cols("final_for_customer_test.csv", attrs_cus_tr, "final_for_customer_test_r.csv")
