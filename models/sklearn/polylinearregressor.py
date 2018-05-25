from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval
import dataset.dataset as ds
import pandas as pd
import numpy as np

# LINEAR REGRESSOR, DEG = 2
# TRAINING SET = mean_var_pre_imputed.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.88122989343
# PREDICTION OF CUSTOMERS : R2 = 0.789715853602

# LINEAR REGRESSOR, DEG = 2
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.886971078082
# PREDICTION OF CUSTOMERS : R2 = 0.847136789595

# Build training & test sets
# data = sb.SetBuilder(target='NumberOfCustomers').exclude('NumberOfSales').exclude('Day').build()
# data = sb.SetBuilder(target='NumberOfSales').exclude('Day').build()
data = sb.SetBuilder(target='NumberOfSales', dataset="fully_preprocessed_ds.csv").build()

poly_degree = 2
NMODELS = 3
models = []

data.ytr = data.ytr.squeeze()
to_pred = data.ytr
poly = PolynomialFeatures(degree=poly_degree)
xtr_ = poly.fit_transform(data.xtr)
xts_ = poly.fit_transform(data.xts)

for i in range(NMODELS):
    # Performs simple linear regression
    print("Linear regression started, polynomial degree = %s" % poly_degree)
    models.append(linear_model.LinearRegression())
    models[i].fit(xtr_, to_pred)
    ypred = models[i].predict(X=xtr_)
    print(ypred)
    to_pred = to_pred - ypred

ypred = np.zeros(shape=[data.yts.shape[0]], dtype=np.float32)

for i in range(NMODELS):
    ypred += models[i].predict(xts_)






ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_LR_DEG2.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))