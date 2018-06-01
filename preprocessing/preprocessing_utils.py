from sklearn import preprocessing as pr
import dataset.dataset as d
import pandas as p
import numpy as np
import preprocessing.imputation as imp


def eliminate_IsOpen_zeros(df):
    df = df[(df[['IsOpen']] != 0).all(axis=1)]
    return df


def add_avg_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    means = dict()
    for id in ids:
        try:
            _ = means[str(id)]
        except KeyError:
            means[str(id)] = average_per_shop(data_from, id)

    df['meanshop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'meanshop', means[str(d.content_of(df, 'StoreID', i))])
    return df


def average_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.mean()


def add_std_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = std_per_shop(data_from, id)

    df['mean_std_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'mean_std_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def std_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.std()


def add_max_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = max_per_shop(data_from, id)

    df['max_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'max_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def max_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.max()


def add_min_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = min_per_shop(data_from, id)

    df['min_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'min_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def min_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfSales']])
    return temp.min()


#############################################


def add_avg_cust_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    means = dict()
    for id in ids:
        try:
            _ = means[str(id)]
        except KeyError:
            means[str(id)] = average_cust_per_shop(data_from, id)

    df['meancustshop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'meancustshop', means[str(d.content_of(df, 'StoreID', i))])
    return df


def average_cust_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfCustomers']])
    return temp.mean()


def add_std_cust_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = std_cust_per_shop(data_from, id)

    df['meancust_std_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'meancust_std_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def std_cust_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfCustomers']])
    return temp.std()


def add_max_cust_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = max_cust_per_shop(data_from, id)

    df['maxcust_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'maxcust_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def max_cust_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfCustomers']])
    return temp.max()


def add_min_cust_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    ids = d.values_of(data_from, 'StoreID')
    stds = dict()
    for id in ids:
        try:
            _ = stds[str(id)]
        except KeyError:
            stds[str(id)] = min_cust_per_shop(data_from, id)

    df['mincust_shop'] = p.Series(np.zeros(len(df)), df.index)
    for i in df.index.tolist():
        df.set_value(i, 'mincust_shop', stds[str(d.content_of(df, 'StoreID', i))])
    return df


def min_cust_per_shop(df, id):
    temp = df[(df[['StoreID']] == id).all(axis=1)]
    temp = d.to_numpy(temp[['NumberOfCustomers']])
    return temp.min()


def mean_std_cust_per_shop_per_day(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanCustPerShopPerDay'] = p.Series(np.zeros(len(df)), df.index)
    df['StdCustPerShopPerDay'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    stds = dict()
    num = dict()
    for i in data_from.index.tolist():
        id = d.content_of(data_from, 'StoreID', i)
        day = d.content_of(data_from, 'Day', i)
        index = str(id) + day
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        id = d.content_of(df, 'StoreID', i)
        day = d.content_of(df, 'Day', i)
        index = str(id) + day
        df.set_value(i, 'MeanCustPerShopPerDay', means[index]/num[index])

    for i in data_from.index.tolist():
        id = d.content_of(data_from, 'StoreID', i)
        day = d.content_of(data_from, 'Day', i)
        index = str(id) + day
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        try:
            stds[index] += (val - means[index]/num[index]) * (val - means[index]/num[index])
        except KeyError:
            stds[index] = (val - means[index]/num[index]) * (val - means[index]/num[index])

    for i in df.index.tolist():
        id = d.content_of(df, 'StoreID', i)
        day = d.content_of(df, 'Day', i)
        index = str(id) + day
        df.set_value(i, 'StdCustPerShopPerDay', np.sqrt(stds[index]/num[index]))
    return df


def mean_std_sales_per_shop_per_day(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanSalesPerShopPerDay'] = p.Series(np.zeros(len(df)), df.index)
    df['StdSalesPerShopPerDay'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    stds = dict()
    num = dict()
    for i in data_from.index.tolist():
        id = d.content_of(data_from, 'StoreID', i)
        day = d.content_of(data_from, 'Day', i)
        index = str(id) + day
        val = d.content_of(data_from, 'NumberOfSales', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        id = d.content_of(df, 'StoreID', i)
        day = d.content_of(df, 'Day', i)
        index = str(id) + day
        df.set_value(i, 'MeanSalesPerShopPerDay', means[index]/num[index])

    for i in data_from.index.tolist():
        id = d.content_of(data_from, 'StoreID', i)
        day = d.content_of(data_from, 'Day', i)
        index = str(id) + day
        val = d.content_of(data_from, 'NumberOfSales', i)
        try:
            stds[index] += (val - means[index]/num[index]) * (val - means[index]/num[index])
        except KeyError:
            stds[index] = (val - means[index]/num[index]) * (val - means[index]/num[index])

    for i in df.index.tolist():
        id = d.content_of(df, 'StoreID', i)
        day = d.content_of(df, 'Day', i)
        index = str(id) + day
        df.set_value(i, 'StdSalesPerShopPerDay', np.sqrt(stds[index]/num[index]))
    return df


def mean_cust_per_month_per_region(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanCustPerRegionPerMonth'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        month = d.content_of(data_from, 'Month', i)
        reg = d.content_of(data_from, 'Region', i)
        index = str(month) + "_" + str(reg)
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        month = d.content_of(df, 'Month', i)
        reg = d.content_of(df, 'Region', i)
        index = str(month) + "_" + str(reg)
        df.set_value(i, 'MeanCustPerRegionPerMonth', means[index]/num[index])
    return df


def mean_cust_per_month_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanCustPerShopPerMonth'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        month = d.content_of(data_from, 'Month', i)
        reg = d.content_of(data_from, 'StoreID', i)
        index = str(month) + "_" + str(reg)
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        month = d.content_of(df, 'Month', i)
        reg = d.content_of(df, 'StoreID', i)
        index = str(month) + "_" + str(reg)
        df.set_value(i, 'MeanCustPerShopPerMonth', means[index]/num[index])
    return df


def mean_cust_per_shop_if_promotions(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanCustPerShopIfPromotions'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        prom = d.content_of(data_from, 'HasPromotions', i)
        reg = d.content_of(data_from, 'StoreID', i)
        index = str(reg)
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        if prom != 0:
            try:
                means[index] += val
                num[index] += 1
            except KeyError:
                means[index] = val
                num[index] = 1
    for i in df.index.tolist():
        prom = d.content_of(df, 'HasPromotions', i)
        reg = d.content_of(df, 'StoreID', i)
        if prom != 0:
            index = str(reg)
            df.set_value(i, 'MeanCustPerShopIfPromotions', means[index]/num[index])
    return df


def mean_cust_per_shop_if_holiday(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanCustPerShopIfHoliday'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        prom = d.content_of(data_from, 'IsHoliday', i)
        reg = d.content_of(data_from, 'StoreID', i)
        index = str(reg)
        val = d.content_of(data_from, 'NumberOfCustomers', i)
        if prom != 0:
            try:
                means[index] += val
                num[index] += 1
            except KeyError:
                means[index] = val
                num[index] = 1
    for i in df.index.tolist():
        prom = d.content_of(df, 'IsHoliday', i)
        reg = d.content_of(df, 'StoreID', i)
        if prom != 0:
            index = str(reg)
            df.set_value(i, 'MeanCustPerShopIfHoliday', means[index]/num[index])
    return df


def mean_sales_per_month_per_shop(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanSalesPerShopPerMonth'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        month = d.content_of(data_from, 'Month', i)
        reg = d.content_of(data_from, 'StoreID', i)
        index = str(month) + "_" + str(reg)
        val = d.content_of(data_from, 'NumberOfSales', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        month = d.content_of(df, 'Month', i)
        reg = d.content_of(df, 'StoreID', i)
        index = str(month) + "_" + str(reg)
        df.set_value(i, 'MeanSalesPerShopPerMonth', means[index]/num[index])
    return df


def mean_sales_per_month_per_region(df, data_from=None):
    if data_from is None:
        data_from = df
    df['MeanSalesPerRegionPerMonth'] = p.Series(np.zeros(len(df)), df.index)
    means = dict()
    num = dict()
    for i in data_from.index.tolist():
        month = d.content_of(data_from, 'Month', i)
        reg = d.content_of(data_from, 'Region', i)
        index = str(month) + "_" + str(reg)
        val = d.content_of(data_from, 'NumberOfSales', i)
        try:
            means[index] += val
            num[index] += 1
        except KeyError:
            means[index] = val
            num[index] = 1
    for i in df.index.tolist():
        month = d.content_of(df, 'Month', i)
        reg = d.content_of(df, 'Region', i)
        index = str(month) + "_" + str(reg)
        df.set_value(i, 'MeanSalesPerRegionPerMonth', means[index]/num[index])
    return df


def add_mean_std_per_shop_per_day(ds):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds = mean_std_cust_per_shop_per_day(ds)
    return ds


def prepare_ds_to_customer_prediction(ds, data_from):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = mean_std_cust_per_shop_per_day(ds, data_from)
    print(ds[['StoreID', 'MeanCustPerShopPerDay', 'StdCustPerShopPerDay']])
    ds = eliminate_IsOpen_zeros(ds)
    ds = add_avg_cust_per_shop(ds, data_from)
    ds = add_std_cust_per_shop(ds, data_from)
    ds = add_max_cust_per_shop(ds, data_from)
    ds = add_min_cust_per_shop(ds, data_from)
    return ds


def prepare_ds_to_sales_prediction(ds, data_from):
    ds['Date'] = p.to_datetime(ds['Date'], format='%d/%m/%Y')
    ds['Day'] = ds['Date'].dt.weekday_name
    ds = imp.one_hot(ds, 'Day', header='Day_')
    ds = mean_std_sales_per_shop_per_day(ds, data_from)
    print(ds[['StoreID', 'MeanSalesPerShopPerDay', 'StdSalesPerShopPerDay']])
    ds = eliminate_IsOpen_zeros(ds)
    ds = add_avg_per_shop(ds, data_from)
    ds = add_std_per_shop(ds, data_from)
    ds = add_max_per_shop(ds, data_from)
    ds = add_min_per_shop(ds, data_from)
    return ds


def full_prep_test_ds_to_sales_pred():
    test_datas = d.read_imputed_onehot_test_dataset()
    data_from = d.read_imputed_onehot_dataset()
    data_from['Date'] = p.to_datetime(data_from['Date'], format='%d/%m/%Y')
    data_from['Day'] = data_from['Date'].dt.weekday_name
    datas = prepare_ds_to_sales_prediction(test_datas, data_from)
    d.save_dataset(datas, "test_dataset_for_sales_prediction.csv")



def full_prep_test_ds_to_cust_pred():
    test_datas = d.read_imputed_onehot_test_dataset()
    data_from = d.read_imputed_onehot_dataset()
    data_from['Date'] = p.to_datetime(data_from['Date'], format='%d/%m/%Y')
    data_from['Day'] = data_from['Date'].dt.weekday_name
    datas = prepare_ds_to_customer_prediction(test_datas, data_from)
    d.save_dataset(datas, "test_dataset_for_customers_prediction.csv")


def reorder_attributes(ds, list_order):
    new = p.DataFrame(index=ds.index)
    for attr in list_order:
        try:
            new[attr] = p.Series(d.to_numpy(ds[[attr]]).squeeze(), index=new.index)
        except Exception:
            new[attr] = p.Series(np.zeros(len(new)), index=new.index)
    return new


def longstring_tolist(string):
    return string.split(",")


def reorder_test_datas_for_cust_pred():
    datas = d.read_dataset("test_dataset_for_customers_prediction.csv")
    datas = reorder_attributes(datas, longstring_tolist("StoreID,Date,IsHoliday,IsOpen,HasPromotions,NearestCompetitor,Region,NumberOfCustomers,NumberOfSales,Region_AreaKM2,Region_GDP,Region_PopulationK,CloudCover,Max_Dew_PointC,Max_Humidity,Max_Sea_Level_PressurehPa,Max_TemperatureC,Max_VisibilityKm,Max_Wind_SpeedKm_h,Mean_Dew_PointC,Mean_Humidity,Mean_Sea_Level_PressurehPa,Mean_TemperatureC,Mean_VisibilityKm,Mean_Wind_SpeedKm_h,Min_Dew_PointC,Min_Humidity,Min_Sea_Level_PressurehPa,Min_TemperatureC,Min_VisibilitykM,Precipitationmm,WindDirDegrees,Events_Rain,Events_Snow,Events_none,Events_Fog,Events_Thunderstorm,Events_Hail,StoreType_Hyper Market,StoreType_Super Market,StoreType_Standard Market,StoreType_Shopping Center,AssortmentType_General,AssortmentType_With Non-Food Department,AssortmentType_With Fish Department,Day,Day_Tuesday,Day_Wednesday,Day_Friday,Day_Saturday,Day_Sunday,Day_Monday,Day_Thursday,MeanCustPerShopPerDay,StdCustPerShopPerDay,meancustshop,meancust_std_shop,maxcust_shop,mincust_shop"))
    d.save_dataset(datas, "test_dataset_for_customers_prediction_reorder.csv")


def reorder_test_datas_for_sales_pred():
    datas = d.read_dataset("test_dataset_for_sales_prediction.csv")
    datas = reorder_attributes(datas, longstring_tolist("StoreID,Date,IsHoliday,IsOpen,HasPromotions,NearestCompetitor,Region,NumberOfCustomers,NumberOfSales,Region_AreaKM2,Region_GDP,Region_PopulationK,CloudCover,Max_Dew_PointC,Max_Humidity,Max_Sea_Level_PressurehPa,Max_TemperatureC,Max_VisibilityKm,Max_Wind_SpeedKm_h,Mean_Dew_PointC,Mean_Humidity,Mean_Sea_Level_PressurehPa,Mean_TemperatureC,Mean_VisibilityKm,Mean_Wind_SpeedKm_h,Min_Dew_PointC,Min_Humidity,Min_Sea_Level_PressurehPa,Min_TemperatureC,Min_VisibilitykM,Precipitationmm,WindDirDegrees,Events_Rain,Events_Snow,Events_none,Events_Fog,Events_Thunderstorm,Events_Hail,StoreType_Hyper Market,StoreType_Super Market,StoreType_Standard Market,StoreType_Shopping Center,AssortmentType_General,AssortmentType_With Non-Food Department,AssortmentType_With Fish Department,Day,Day_Tuesday,Day_Wednesday,Day_Friday,Day_Saturday,Day_Sunday,Day_Monday,Day_Thursday,MeanSalesPerShopPerDay,StdSalesPerShopPerDay,meanshop,mean_std_shop,max_shop,min_shop"))
    d.save_dataset(datas, "test_dataset_for_sales_prediction_reorder.csv")


if __name__ == '__main__':
    reorder_test_datas_for_sales_pred()