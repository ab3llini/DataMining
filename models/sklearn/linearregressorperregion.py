import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import evaluation.evaluation as ev_cust
from preprocessing import preprocessing_utils
import dataset.dataset as ds
import dataset.utility as utils
from sklearn import linear_model
from sklearn import ensemble
from sklearn import pipeline
from sklearn import tree
import numpy as np
import pandas
import models.sklearn.sklearnlinearclass as skc
from sklearn.preprocessing import PolynomialFeatures

def prepare_out(df):
    y = ds.to_numpy(df[['NumberOfCustomers']]).squeeze()
    return y


def drop_useless(df, axis=0):
    x = ds.to_numpy(df.drop(['NumberOfSales', 'StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover',
                            'Max_Sea_Level_PressurehPa', 'WindDirDegrees', 'Max_Dew_PointC',
                            'NumberOfCustomers', 'Day', 'Mean_Sea_Level_PressurehPa',
                            'Min_Sea_Level_PressurehPa'], axis=axis))
    return x


def model():
    return ensemble.BaggingRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=2), n_estimators=150, bootstrap=True)


datas = ds.read_dataset("mean_var_on_customers_from_tain.csv")
train = utils.get_frame_in_range(datas, 3, 2016, 12, 2017)
test = utils.get_frame_in_range(datas, 1, 2018, 2, 2018)
regions_n = 11
poly = PolynomialFeatures(degree=1)
sum = 0
models = []
for i in range(regions_n):
    print("REG " + str(i))
    d_reg = utils.get_frames_per_region(train, i)
    d_reg_t = utils.get_frames_per_region(test, i)
    print("N_SAMPLES: ", len(d_reg) + len(d_reg_t))
    y = prepare_out(d_reg)
    x = drop_useless(d_reg, 1)
    y_t = prepare_out(d_reg_t)
    x_t = drop_useless(d_reg_t, 1)
    mod = skc.LinearSklearn(1, model)
    mod.train(x, y)
    models.append(mod)
    p = mod.predict(x).squeeze()
    pt = mod.predict(x_t).squeeze()
    r2_t = eval.r2_score(y_t, pt)
    sum += r2_t * (len(d_reg) + len(d_reg_t))
    print("TRAIN R2: ", eval.r2_score(y, p))
    print("TEST R2: ", r2_t)
    print("##########################")

print("AVG TEST R2: ", sum/len(datas))

custpred = []
for i in test.index.tolist():
    row = test.loc[i]
    reg = row['Region']
    row = drop_useless(row).reshape([1, -1])
    custpred.append(models[reg].predict(row).squeeze())

custpred = np.array(custpred)

new = pandas.DataFrame(index=test.index)
new['NumberOfCustomers'] = pandas.Series(custpred, test.index)
ds.save_dataset(new, "cust_ensemble_per_region_predictions1.csv")