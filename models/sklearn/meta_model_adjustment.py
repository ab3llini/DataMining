import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import evaluation.evaluation as ev_cust
from preprocessing import preprocessing_utils
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


def ridge():
    return linear_model.Ridge(alpha=5)


def lasso():
    return linear_model.Lasso(alpha=5)


def regtree():
    return tree.DecisionTreeRegressor(max_depth=9)


def gradboostreg():
    return ensemble.GradientBoostingRegressor(max_depth=8, n_estimators=5)


if __name__ == '__main__':
    datas = ds.read_dataset("mean_var_on_customers_from_tain.csv")
    datas['Month'] = datas['Date']
    datas['Month'] = datas['Month'].apply(lambda x: x.split("-")[1])
    datas = imp.one_hot_numeric(datas, 'Month', 'Month_')
    datas = imp.one_hot_numeric(datas, 'Region', 'Region_')
    datas = preprocessing_utils.mean_cust_per_month_per_region(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    datas = preprocessing_utils.mean_cust_per_month_per_shop(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas)\
        .exclude('NumberOfSales', 'Month', 'Max_Humidity', 'Max_Sea_Level_PressurehPa', 'Max_TemperatureC',
                 'Max_VisibilityKm', 'Max_Wind_SpeedKm_h', 'Mean_Humidity', 'Mean_Sea_Level_PressurehPa',
                 'Mean_VisibilityKm', 'Mean_Wind_SpeedKm_h', 'Min_Dew_PointC',
                 'Min_Humidity', 'Min_Sea_Level_PressurehPa', 'Min_TemperatureC', 'Min_VisibilitykM')\
        .build()
    model =[ridge, linear_model.LinearRegression, lasso, regtree, gradboostreg]
    final = ridge
    n = len(model)
    mods = []
    modpreds = []
    modpreds_t = []
    for i in range(n):
        print(i+1)
        x, y = datas.xtr, datas.ytr
        mod = skc.LinearSklearn(1, model[i])
        mod.train(x, y)
        mods.append(mod)
        p = mod.predict(x).squeeze()
        print(p)
        p_t = mod.predict(datas.xts).squeeze()
        modpreds.append(p)
        modpreds_t.append(p_t)
        print("TRAIN R2: ", eval.r2_score(y, p))
        print("TEST R2: ", eval.r2_score(datas.yts, p_t))
        print("##########################")

    modpreds = np.array(modpreds).transpose()
    modpreds_t = np.array(modpreds_t).transpose()
    x = np.hstack((datas.xtr, modpreds))
    x_t = np.hstack((datas.xts, modpreds_t))

    fin = skc.LinearSklearn(1, final)
    fin.train(x, datas.ytr)
    custpred = fin.predict(x_t)
    print(custpred)
    print("TEST R2: ", eval.r2_score(datas.yts, custpred))

    new = pandas.DataFrame()
    new['NumberOfCustomers'] = pandas.Series(custpred)
    ds.save_dataset(new, "meta_mod2.csv")
