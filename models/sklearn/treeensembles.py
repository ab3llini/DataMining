from sklearn.linear_model import Ridge

import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import evaluation.evaluation as ev_cust
from preprocessing import preprocessing_utils as preu
from sklearn.pipeline import make_pipeline
import dataset.dataset as ds
import dataset.utility as utils
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
import numpy as np
from inspect import getmembers
import preprocessing.imputation as imp
import pandas
import models.sklearn.sklearnlinearclass as skc
from sklearn.preprocessing import PolynomialFeatures


def excluded_feats():
    return ["Month", 'Max_Humidity','Max_TemperatureC','Max_VisibilityKm',
            'Max_Wind_SpeedKm_h','Min_Dew_PointC','Min_Humidity','Min_TemperatureC','Min_VisibilitykM',
            'NumberOfSales']


def model():
    return tree.DecisionTreeRegressor(max_depth=9)

# split=[[(3, 2016, 2, 2017), (5, 2017, 2, 2018)], [(3, 2017, 4, 2017)]]


def build_cust_predictor_train_dataset(m1, a1, m2, a2):
    das = ds.read_imputed_onehot_dataset()
    das = __prepare_customers_train_ds(das, m1, a1, m2, a2)
    return das


def __prepare_customers_train_ds(das, m1, a1, m2, a2):
    das['Date'] = pandas.to_datetime(das['Date'], format='%d/%m/%Y')
    das['Day'] = das['Date'].dt.weekday_name
    das['Date'] = das['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    das['Month'] = das['Date']
    das['Month'] = das['Month'].apply(lambda x: x.split("-")[1])
    das = imp.one_hot(das, 'Day', header='Day_')
    das = imp.one_hot_numeric(das, 'Month', 'Month_')
    das = imp.one_hot_numeric(das, 'Region', 'Region_')
    dfrom = utils.get_frame_out_of_range(das, m1, a1, m2, a2)
    das = preu.eliminate_IsOpen_zeros(das)
    das = preu.mean_std_cust_per_shop_per_day(das, dfrom)
    das = preu.add_avg_cust_per_shop(das, dfrom)
    das = preu.add_std_cust_per_shop(das, dfrom)
    das = preu.add_max_cust_per_shop(das, dfrom)
    das = preu.add_min_cust_per_shop(das, dfrom)
    das = preu.mean_cust_per_month_per_shop(das, dfrom)
    das = preu.mean_cust_per_month_per_region(das, dfrom)
    return das


if __name__ == '__main__':
    split = [[(3, 2016, 12, 2016), (3, 2017, 2, 2018)], [(1, 2017, 2, 2017)]]
    datas = build_cust_predictor_train_dataset(1, 2017, 2, 2017)
    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas, split=split)\
        .exclude("Month", "NumberOfSales")\
        .build()
    n = 10
    mods = []
    for i in range(n):
        print(i+1)
        x, y = datas.random_sampling(1.0)
        mod = skc.LinearSklearn(1, model)
        mod.train(x, y)
        mods.append(mod)
        p = mod.predict(datas.xtr).squeeze()
        print("TRAIN R2: ", eval.r2_score(datas.ytr, p))
        print("TEST R2: ", eval.r2_score(datas.yts, mod.predict(datas.xts)))
        print("##########################")


    preds = []
    for i in range(n):
        preds.append(mods[i].predict(datas.xts))

    custpred = np.array(preds).mean(axis=0)

    print("TEST R2: ", eval.r2_score(datas.yts, custpred))

    new = pandas.DataFrame()
    new['NumberOfCustomers'] = pandas.Series(custpred)
    ds.save_dataset(new, "genfeb17.csv")
