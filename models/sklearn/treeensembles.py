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

def prepare_out(df):
    y = ds.to_numpy(df[['NumberOfCustomers']]).squeeze()
    return y


def drop_useless(df, axis=0):
    dropped = df.drop(['NumberOfSales', 'StoreID', 'Date', 'IsOpen', 'CloudCover',
                            'Max_Sea_Level_PressurehPa', 'WindDirDegrees', 'Max_Dew_PointC',
                            'NumberOfCustomers', 'Day', 'Mean_Sea_Level_PressurehPa',
                            'Min_Sea_Level_PressurehPa', 'Region', 'Month'], axis=axis)
    #print(list(dropped))
    x = ds.to_numpy(dropped)
    return x


def model():
    return linear_model.Ridge(alpha=5)


if __name__ == '__main__':
    datas = ds.read_dataset("mean_var_on_customers_from_tain.csv")
    datas['Month'] = datas['Date']
    datas['Month'] = datas['Month'].apply(lambda x: x.split("-")[1])
    datas = imp.one_hot_numeric(datas, 'Month', 'Month_')
    datas = imp.one_hot_numeric(datas, 'Region', 'Region_')
    datas = preprocessing_utils.mean_cust_per_month_per_region(datas, utils.get_frame_in_range(datas, 3, 2016, 12, 2017))
    datas = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datas).exclude('NumberOfSales', 'Month').build()
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

    #tree.export_graphviz(mod.models[0])
    #print("SAVED")
    preds = []
    for i in range(n):
        preds.append(mods[i].predict(datas.xts))

    custpred = np.array(preds).mean(axis=0)

    print("TEST R2: ", eval.r2_score(datas.yts, custpred))

    new = pandas.DataFrame()
    new['NumberOfCustomers'] = pandas.Series(custpred)
    ds.save_dataset(new, "cust_ensemble_predictions6.csv")
