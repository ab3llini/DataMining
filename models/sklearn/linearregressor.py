import sklearn as skl
from sklearn import linear_model
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval
import evaluation.evaluation as ev_cust
import dataset.dataset as ds
import dataset.utility as utils
import numpy as np
import models.sklearn.sklearnlinearclass as skc

# LINEAR REGRESSOR, DEG = 1
# TRAINING SET = mean_var_pre_imputed.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.845990358862
# PREDICTION OF CUSTOMERS : R2 = 0.811300139668

# LINEAR REGRESSOR, DEG = 1
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.847824705349
# PREDICTION OF CUSTOMERS : R2 = 0.86925841471

# Build training & test sets

# data = sb.SetBuilder(target='NumberOfCustomers').exclude('NumberOfSales').exclude('Day').build()
# data = sb.SetBuilder(target='NumberOfSales', dataset="fully_preprocessed_ds.csv").build()
ds_name = "fully_preprocessed_ds.csv"
NMODELS = 3
SET_OF_MODELS_DIM = 1
models = []
datas = ds.read_dataset(ds_name)
datas = utils.get_frame_in_range(datas, 1, 2018, 2, 2018)
regions = ds.to_numpy(datas[['Region']]).squeeze()
dates = ds.to_numpy(datas[['Date']]).squeeze()
ids = ds.to_numpy(datas[['StoreID']]).squeeze()

def mod():
    return skl.linear_model.LinearRegression()


for i in range(SET_OF_MODELS_DIM):
    data = sb.SetBuilder(target='NumberOfSales', dataset=ds_name).build()
    data.random_sampling(1.0)
    model = skc.LinearSklearn(NMODELS, mod)
    # Performs simple linear regression
    model.train(data.xtr, data.ytr)
    models.append(model)

data = sb.SetBuilder(target='NumberOfSales', dataset=ds_name).build()
ypred = np.zeros(shape=data.yts.shape[0])
for i in range(SET_OF_MODELS_DIM):
    ypred += models[i].predict(data.xts)

ypred = ypred/SET_OF_MODELS_DIM

print('R2 = %s' % eval.evaluate(data.yts, ypred))

re, totr, totp = ev_cust.region_error(data.yts, ypred, regions, ids, dates)
diff = totr - totp
print(re)
print(re.mean())
print(totr)
print(diff.sum())
print(diff)
