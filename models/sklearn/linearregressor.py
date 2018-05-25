from sklearn import linear_model
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval

# LINEAR REGRESSOR, DEG = 1
# TRAINING SET = mean_var_pre_imputed.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.845990358862
# PREDICTION OF CUSTOMERS : R2 = 0.811300139668

# LINEAR REGRESSOR, DEG = 1
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.847824705349
# PREDICTION OF CUSTOMERS : R2 = 0.86925841471

# Build training & test sets
data = sb.SetBuilder(target='NumberOfSales', default=False).exclude('Day').build()
# data = sb.SetBuilder(target='NumberOfSales').exclude('Day').build()

# Performs simple linear regression
print("Linear regression started, polynomial degree = 1")

regression = linear_model.LinearRegression()
regression.fit(data.xtr, data.ytr)
ypred = regression.predict(data.xts)

# ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_LR_DEG1.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))
