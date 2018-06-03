from sklearn.linear_model import Ridge

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


def model():
    return linear_model.Lasso(alpha=5)


if __name__ == '__main__':
    datas = ds.read_dataset("best_for_customers.csv")
    cols = list(datas)
    cols.remove('NumberOfSales')
    cols.remove('Month')
    cols.remove('NumberOfCustomers')
    for att in ['StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover', 'Max_Sea_Level_PressurehPa', 'WindDirDegrees',
                'Max_Dew_PointC', 'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa', 'Day']:

        cols.remove(att)
    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas)\
        .exclude('NumberOfSales', 'Month')\
        .build()
    n = 1
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

    #tree.export_graphviz(mod.models[0])
    #print("SAVED")
    preds = []
    for i in range(n):
        preds.append(mods[i].predict(datas.xts))

    custpred = np.array(preds).mean(axis=0)

    print("TEST R2: ", eval.r2_score(datas.yts, custpred))
    print("############################################")
    for i in range(n):
        for j in range(len(cols)):
            print(cols[j], " : ", mods[i].models[0].coef_[j])
        print("###############################################")
