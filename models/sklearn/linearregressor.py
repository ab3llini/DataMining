from sklearn import linear_model
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval

# LINEAR REGRESSOR, DEG = 1
# TRAINING SET = mean_var_pre_imputed.csv
# PREDICTION OF SALES (with customers as input) : R2 = 0.908379273457
# PREDICTION OF CUSTOMERS : R2 = 0.832673686457

# Build training & test sets
data = sb.SetBuilder(target='NumberOfCustomers').exclude('NumberOfSales').build()

# Performs simple linear regression
print("Linear regression started, polynomial degree = 1")

regression = linear_model.LinearRegression()
regression.fit(data.xtr, data.ytr)
ypred = regression.predict(data.xts)

print('R2 = %s' % eval.evaluate(data.yts, ypred))
