from sklearn import tree
import dataset.dataset as ds
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval


# Build training & test sets
data = sb.SetBuilder(target='NumberOfSales', autoexlude=True).build()

# Performs simple linear regression
print("Decision regression tree")

dtree = tree.DecisionTreeRegressor()
dtree.fit(data.xtr, data.ytr)
ypred = dtree.predict(data.xts)

# ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_LR_DEG1.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))

