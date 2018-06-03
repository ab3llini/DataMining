from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval


data = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True).exclude('NumberOfSales', 'Month').build()


model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.2,
    colsample_bytree=1,
    max_depth=4,
    silent=False,
    n_jobs=8
)

model.fit(data.xtr, data.ytr)

pred_tr = model.predict(data.xtr)
pred_ts = model.predict(data.xts)

print('R2 TRAIN = %s' % eval.evaluate(data.ytr, pred_tr))
print('R2 TEST = %s' % eval.evaluate(data.yts, pred_ts))
