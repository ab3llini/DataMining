from sklearn import tree
import models.sklearn.setbuilder as sb
import models.sklearn.evaluator as eval
import dataset.dataset as ds
import pandas as pd

# Build training & test sets
data = sb.SetBuilder(target='NumberOfSales').build()

# Performs simple linear regression
print("Decision regression tree")

regression = tree.DecisionTreeRegressor()
regression.fit(data.xtr, data.ytr)
ypred = regression.predict(data.xts)

# ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_LR_DEG1.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))