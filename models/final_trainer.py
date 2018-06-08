from sklearn import linear_model
import models.sklearn.sklearnlinearclass as skc
import models.sklearn.evaluator as eval
import numpy as np
import dataset.setbuilder as sb
import models.sklearn.persistence as persistence
import pandas as pd
import dataset.dataset as ds
from sklearn import tree


class Bagging:

    def __init__(self, count):

        self.count = count
        self.predictions = []
        self.training_predictions = []
        self.description = ""
        self.final_predictions = []
        self.final_training_predictions = []

    def compute_final_predictions(self):
        print("Computing average predictions for bagging on customers..")
        self.final_predictions = np.array(self.predictions).mean(axis=0)
        self.final_training_predictions = np.array(self.training_predictions).mean(axis=0)

    def save_partial(self, model, it):
        model_name = "%s_bagging_partial_%s" % (self.description, it)
        persistence.save_model(model, model_name)
        print("Model %s :: Customer partial model saved as %s" % (it, model_name))


class SalesModel(Bagging):

    # We must provide a target dataset with the predicted customers
    def __init__(self, count,  target):
        Bagging.__init__(self, count)

        print("Remember: IS IS MANDATORY THAT THE COLUMN ORDER OF TARGET SET IS THE SAME OF final_for_sales_train.csv")

        self.description = "sales"

        self.target = target

        self.training_ds = sb.SetBuilder(
            target='NumberOfSales',
            autoexclude=True,
            split=[[(3, 2016, 2, 2018)], []],
            dataset="final_for_sales_train.csv"
        ).exclude('Month').build()

        self.testing_ds = sb.SetBuilder(
            target='NumberOfSales',
            autoexclude=True,
            split=[[], [(3, 2018, 4, 2018)]],
            df=target
        ).exclude('Month').build()

    def fit(self):
        for i in range(self.count):
            x_sampled, y_sampled = self.training_ds.random_sampling(1.0)
            curr = tree.DecisionTreeRegressor()
            curr.fit(x_sampled, y_sampled)
            tr_predictions = curr.predict(self.training_ds.xtr)

            self.training_predictions.append(tr_predictions)

            print("Model %s :: Sales R2 on training set: %s" % (i, eval.r2_score(self.training_ds.ytr, tr_predictions)))

            ts_predictions = curr.predict(self.testing_ds.xts)
            self.predictions.append(ts_predictions)

            # We cannot evaluate our predictions when building the final model, we don't have a test set :)
            # print("Customer R2 on testing set: ", eval.r2_score(self.ds.yts, ts_predictions))
            self.save_partial(curr, i)

        self.compute_final_predictions()

        print("Overall sales R2 on training set: %s" % eval.r2_score(self.training_ds.ytr, self.final_training_predictions))

        print("Done with sales bagging, models have been saved in 'saved' dir.")
        print("When ready, execute save_predictions.")

    # Saves and returns the data frame containing the predictions
    def save_predictions(self, header="NumberOfSales", csv="final_sales_predictions.csv"):
        new = pd.DataFrame()
        new[header] = pd.Series(self.final_predictions)
        ds.save_dataset(new, csv)
        return new


class CustomerModel(Bagging):

    def __init__(self, count):
        Bagging.__init__(self, count)

        self.description = "customers"

        self.training_ds = sb.SetBuilder(
            target='NumberOfCustomers',
            autoexclude=True,
            split=[[(3, 2016, 2, 2018)], []],
            dataset="final_for_customer_train.csv"
        ).exclude('NumberOfSales', 'Month').build()

        self.testing_ds = sb.SetBuilder(
            target='NumberOfCustomers',
            autoexclude=True,
            split=[[], [(3, 2018, 4, 2018)]],
            dataset="final_for_customer_test_r.csv"
        ).exclude('NumberOfSales', 'Month').build()

    @staticmethod
    def model():
        return linear_model.Ridge(alpha=200)

    def fit(self):

        for i in range(self.count):

            x_sampled, y_sampled = self.training_ds.random_sampling(1.0)
            curr = skc.LinearSklearn(1, CustomerModel.model)
            curr.train(x_sampled, y_sampled)
            tr_predictions = curr.predict(self.training_ds.xtr).squeeze()

            self.training_predictions.append(tr_predictions)

            print("Model %s :: Customer R2 on training set: %s" % (i, eval.r2_score(self.training_ds.ytr, tr_predictions)))

            ts_predictions = curr.predict(self.testing_ds.xts)
            self.predictions.append(ts_predictions)

            # We cannot evaluate our predictions when building the final model, we don't have a test set :)
            # print("Customer R2 on testing set: ", eval.r2_score(self.ds.yts, ts_predictions))
            self.save_partial(curr, i)

        self.compute_final_predictions()

        print("Overall customer R2 on training set: %s" % eval.r2_score(self.training_ds.ytr, self.final_training_predictions))

        print("Done with customer bagging, models have been saved in 'saved' dir.")
        print("When ready, execute save_predictions.")

    # Saves and returns the data frame containing the predictions
    def save_predictions(self, header="NumberOfCustomers", csv="final_customer_predictions.csv"):
        new = pd.DataFrame()
        new[header] = pd.Series(self.final_predictions)
        ds.save_dataset(new, csv)
        return new


'''
Logic of the final trainer:

STAGE ONE
1. Create a new instance of CustomerModel
2. Fit CustomerModel with the whole final training set for the customers
3. Obtain the predicted customers

STAGE TWO
1. inject the predicted customers in the TESTING set for the sales

STAGE THREE
1. Create a new instance of SalesModel
2. When creating the new instance, pass a target set containing the testing set for the sales with injected customers
3. Fit SalesModel with the whole final training set for the sales. 
4. Obtain the predicted sales with the model trained on training set + predicted customers on test set

'''

print("******* Beginning final training & prediction")

csv_cust = "final_customer_predictions.csv"
csv_sales = "final_sales_predictions.csv"

bagging_count_cust = 10
bagging_count_sales = 10


# Stage 1 : Customer prediction
print("******* Beginning stage 1.")
customers = CustomerModel(count=bagging_count_cust)
customers.fit()
predicted_customers = customers.save_predictions(header="NumberOfCustomers", csv=csv_cust)
print("******* Done stage 1.")

# Stage 2: Injection of customers into test set
print("******* Beginning stage 2")
target = ds.read_dataset(name="final_for_sales_test_r.csv")
target["NumberOfCustomers"] = predicted_customers
print("******* Done stage 2.")

# Stage 3
print("******* Beginning stage 3")
sales = SalesModel(count=bagging_count_sales, target=target)
sales.fit()
predicted_sales = sales.save_predictions(header="NumberOfSales", csv=csv_sales)
print("******* Done stage 3.")


print("******* All done. Bagging partials are saved in /saved dir, final predictions in /dataset dir.")

