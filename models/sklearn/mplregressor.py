from sklearn import neural_network
import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import dataset.dataset as ds
import pandas as pd



# MLP REGRESSOR (100,5)
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.914437253913
# PREDICTION OF CUSTOMERS : R2 = 0.887600550772

# MLP REGRESSOR (80,5)
# TRAINING SET = mean_var_pre_imputed_per_day.csv
# CLOSED STORES ARE NOT CONSIDERED
# PREDICTION OF SALES (with customers as input) : R2 = 0.911611329049
# PREDICTION OF CUSTOMERS : R2 = 0.890258164994


# Build training & test sets
data = sb.SetBuilder(target='NumberOfCustomers').exclude('NumberOfSales').exclude('Day').build()
#data = sb.SetBuilder(target='NumberOfSales').exclude('Day').build()

nn = neural_network.MLPRegressor(
    hidden_layer_sizes=(80,5),
    activation='relu',
    solver='adam',
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=50,
    shuffle=True,
    random_state=9,
    tol=0.000001,
    verbose=True,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
)

n = nn.fit(data.xtr, data.ytr.ravel())

ypred = nn.predict(data.xts)

ds.save_dataset(pd.DataFrame(ypred), 'customer_pred_jan_feb_NN.csv')

print('R2 = %s' % eval.evaluate(data.yts, ypred))