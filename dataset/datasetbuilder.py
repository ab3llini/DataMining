import dataset.dataset as d
import preprocessing.preprocessing_utils as pre_u
import pandas as p
import preprocessing.imputation as imp


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
    return ds