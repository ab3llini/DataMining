from sklearn import tree
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import numpy as np
import models.sklearn.persistence as pr


print("Plain Decision regression tree without bagging")

# Build training & test sets
# data = sb.SetBuilder(target='NumberOfCustomers', autoexlude=True).exclude('NumberOfSales').build()
data = sb.SetBuilder(target='NumberOfSales', autoexclude=True, dataset='mean_var_on_sales_from_tain.csv').build()

# Performs simple linear regression

dtree = tree.DecisionTreeRegressor()
dtree.fit(data.xtr, data.ytr)
ypred = dtree.predict(data.xts)

pr.save_model(dtree, 'decision_tree_sales')

dtree = pr.load_model('decision_tree_sales.gz')
ypred = dtree.predict(data.xts)


print('R2 = %s' % eval.evaluate(data.yts, ypred))
print("Plain Decision regression tree without bagging")

it = 100
yy = []
for i in range(it):
    bagx, bagy = data.random_sampling(0.65)
    dt = tree.DecisionTreeRegressor()
    dt.fit(bagx, bagy)
    pr.save_model(dt, 'dt_sales_bagging_%s' % i)
    y = dt.predict(data.xts)
    print('it = %s, R2 = %s' % (i, eval.evaluate(data.yts, y)))
    yy.append(y)

yy = np.array(yy)

pred = yy.mean(axis=0)

for i, e in enumerate(pred):
    print("(prediction) %s : %s (actual)" % (e, data.yts[i]))

print('Bagging R2 = %s' % eval.evaluate(data.yts, pred))





