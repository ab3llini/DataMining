from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import dataset.setbuilder as sb

import models.sklearn.evaluator as eval
import evaluation.evaluation as ev_cust
import dataset.dataset as ds
import dataset.utility as utils
import numpy as np


data = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True).exclude('NumberOfSales', 'Month').build()

deg_1_mod = linear_model.Ridge(5)
deg_2_mod = make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge(5))

lr = 0.3

deg_1_mod.fit(data.xtr, data.ytr)
deg_1_predictions = deg_1_mod.predict(data.xtr)
to_correct = (data.ytr - deg_1_predictions) * lr
deg_2_mod.fit(data.xtr, to_correct)
correction = deg_2_mod.predict(data.xtr)
corrected = correction + deg_1_predictions

print('R2 TRAIN = %s' % eval.evaluate(data.ytr, corrected))

deg_1_predictions = deg_1_mod.predict(data.xts)
correction = deg_2_mod.predict(data.xts)
corrected = correction + deg_1_predictions

print('R2 TEST = %s' % eval.evaluate(data.yts, corrected))
