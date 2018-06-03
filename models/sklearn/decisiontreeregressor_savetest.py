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


def model():
    return tree.DecisionTreeRegressor(max_depth=7)


def corr():
    return linear_model.Ridge()


def printcols():
    cols = list(datas)
    cols.remove('NumberOfSales')
    cols.remove('Month')
    cols.remove('NumberOfCustomers')
    for att in ['StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover', 'Max_Sea_Level_PressurehPa', 'WindDirDegrees',
                'Max_Dew_PointC', 'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa', 'Day']:
        cols.remove(att)
    print(cols)


if __name__ == '__main__':
    datas = ds.read_dataset("best_for_customers.csv")
    datas = preu.mean_cust_per_shop_if_promotions(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    datas = preu.mean_cust_per_shop_if_holiday(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas)\
        .exclude('NumberOfSales', 'Month')\
        .build()
    n = 10
    mods = []
    for i in range(n):
        print(i+1)
        x, y = datas.random_sampling(1.0)
        mod = skc.LinearSklearn(1, model)
        mod.train(x, y)
        mods.append(mod)
        p = mod.predict(x).squeeze()
        print("TRAIN R2: ", eval.r2_score(y, p))
        print("TEST R2: ", eval.r2_score(datas.yts, mod.predict(datas.xts)))
        print("##########################")

    preds = []
    for i in range(n):
        preds.append(mods[i].predict(datas.xtr))

    trainpreds = np.array(preds).mean(axis=0).squeeze()

    preds = []
    for i in range(n):
        preds.append(mods[i].predict(datas.xts))

    custpred = np.array(preds).mean(axis=0).squeeze()

    print("TREE TEST R2: ", eval.r2_score(datas.yts, custpred))
    # tree.export_graphviz(mod.models[0])

    correction = []
    for i in range(len(datas.ytr)):
        correction.append(datas.ytr[i] - trainpreds[i])
    correction = np.array(correction)

    mod = skc.LinearSklearn(1, corr)
    mod.train(datas.xtr, correction)
    p = mod.predict(datas.xtr).squeeze()
    pt = mod.predict(datas.xts).squeeze()

    print("FINAL TRAIN R2: ", eval.r2_score(datas.ytr, trainpreds + p))
    print("FINAL TEST R2: ", eval.r2_score(datas.yts, custpred + pt))

