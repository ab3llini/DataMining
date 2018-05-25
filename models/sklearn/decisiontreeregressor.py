from sklearn import tree
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval

print("Plain Decision regression tree without bagging")

# Build training & test sets
data = sb.SetBuilder(target='NumberOfSales', autoexlude=True).build()

# Performs simple linear regression

dtree = tree.DecisionTreeRegressor()
dtree.fit(data.xtr, data.ytr)
ypred = dtree.predict(data.xts)

print('R2 = %s' % eval.evaluate(data.yts, ypred))

