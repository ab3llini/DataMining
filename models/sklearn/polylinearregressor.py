
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval

# LINEAR REGRESSOR, DEG = 2
# TRAINING SET = mean_var_pre_imputed.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.88122989343
# PREDICTION OF CUSTOMERS : R2 = 0.789715853602

# LINEAR REGRESSOR, DEG = 2
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.886971078082
# PREDICTION OF CUSTOMERS : R2 = 0.847136789595

# Build training & test sets
data = sb.SetBuilder(target='NumberOfSales', default=False).only(['NearestCompetitor', 'Region_AreaKM2', 'NumberOfCustomers', 'CustomersMeanPerStore' ,'CustomersVariancePerStore','MeanCustPerShopPerDay', 'StdCustPerShopPerDay']).build()
# data = sb.SetBuilder(target='NumberOfSales').exclude('Day').build()

poly_degree = 5

# Performs simple linear regression
print("Linear regression started, polynomial degree = %s" % poly_degree)
poly = PolynomialFeatures(degree=poly_degree)
xtr_ = poly.fit_transform(data.xtr)
xts_ = poly.fit_transform(data.xts)

print("Fit transformed")

clf = linear_model.Lasso()
clf.fit(xtr_, data.ytr)

print("Fit done")

ypred = clf.predict(xts_)


#ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_LR_DEG2.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))