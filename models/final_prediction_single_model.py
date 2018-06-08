from sklearn import tree
import dataset.setbuilder as sb
import dataset.dataset as d
import models.sklearn.evaluator as eval
import numpy as np
import evaluation.evaluation as ev_cust
import dataset.utility as utils
import pandas
import sklearn.neural_network as nn


def excluded_feats():
    return ["Month", 'Max_Humidity','Max_TemperatureC','Max_VisibilityKm',
           'Max_Wind_SpeedKm_h','Min_Dew_PointC','Min_Humidity','Min_TemperatureC',
            'Min_VisibilitykM', 'NumberOfCustomers']


datas = d.read_dataset("final_sales_only_train.csv")
regions = d.to_numpy(datas[['Region']]).squeeze()
dates = d.to_numpy(datas[['Date']]).squeeze()
ids = d.to_numpy(datas[['StoreID']]).squeeze()

data = sb.SetBuilder(target='NumberOfSales', autoexclude=True, df=datas.copy(), split=(3, 2016, 2, 2018, 12, 2018, 12, 2018))\
    .exclude_list(excluded_feats())\
    .build()


it = 1
yy = []
models = []
for i in range(it):
    bagx, bagy = data.xtr, data.ytr
    dt = nn.MLPRegressor(hidden_layer_sizes=(400,3),
                            activation='identity',
                            solver='adam',
                            batch_size=50000,
                            learning_rate='adaptive',
                            learning_rate_init=0.002,
                            max_iter=50,
                            shuffle=True,
                            tol=0.000001,
                            verbose=True,
                            warm_start=False,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            early_stopping=False,
                            validation_fraction=0.1,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-08)
    dt.fit(bagx, bagy.ravel())
    models.append(dt)
    y = dt.predict(data.xtr)
    print('it = %s, TRAIN R2 = %s' % (i, eval.evaluate(data.ytr, y)))
    yy.append(y)

yy = np.array(yy)

pred = yy.mean(axis=0)

print('Bagging R2 = %s' % eval.evaluate(data.ytr, pred))


re, totr, totp = ev_cust.region_error(data.ytr, pred, regions, ids, dates)
diff = totr - totp
print("REG_ERR: ", re * 100)
print("REG_MEAN_ERR: ", re.mean() * 100)
print("REAL_SUM: ", totr.sum())
print("PRED_SUM: ", totp.sum())
print("SUM_OF_DIFFS: ", diff.sum())


data_t = d.read_dataset("final_sales_only_test_r.csv")
data_t = sb.SetBuilder(target='NumberOfSales', autoexclude=True, df=data_t.copy(),
                       split=(3, 2016, 2, 2018, 3, 2018, 4, 2018))\
    .exclude_list(excluded_feats())\
    .build()

yy = []
for i in range(it):
    y = models[i].predict(data_t.xts)
    yy.append(y)

pred = np.array(yy).mean(axis=0)

tosave = pandas.DataFrame()
tosave['NumberOfSales'] = pandas.Series(pred)
d.save_dataset(tosave, "sales_only_final_pred_nn.csv")