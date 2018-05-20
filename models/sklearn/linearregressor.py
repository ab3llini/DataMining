from sklearn import linear_model
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval

# Build training & test sets
data = sb.SetBuilder(target='NumberOfCustomers').exclude('NumberOfSales').build()

# Performs simple linear regression
print("Linear regression started, polynomial degree = 1")

regression = linear_model.LinearRegression()
regression.fit(data.xtr, data.ytr)
ypred = regression.predict(data.xts)

print('R2 = %s' % eval.evaluate(data.yts, ypred))
