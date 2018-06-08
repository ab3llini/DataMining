import dataset.utility as dsutil
import dataset.dataset as ds
import seaborn as sea
import dataset.setbuilder as sb
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models.sklearn.evaluator as eval
from sklearn.preprocessing import PolynomialFeatures


data = sb.SetBuilder(
    target="NumberOfSales",
    dataset="final_for_sales_train.csv",
    autoexclude=False,
    split=[[(3, 2016, 1, 2018)], [(3, 2016, 2, 2018)]]
).only('NearestCompetitor').build()


poly_degree = 2

# Performs simple linear regression
print("Linear regression started, polynomial degree = %s" % poly_degree)
poly = PolynomialFeatures(degree=poly_degree)
xtr_ = poly.fit_transform(data.xtr)
xts_ = poly.fit_transform(data.xts)

model = linear_model.LinearRegression()

model.fit(data.xtr, data.ytr)

print(eval.evaluate(data.ytr, model.predict(data.xtr)))
print(eval.evaluate(data.yts, model.predict(data.xts)))

print(model.coef_)
print(model.intercept_)