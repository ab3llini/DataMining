import dataset.setbuilder as sb
import models.sklearn.evaluator as eval
import dataset.dataset as data_manager
from sklearn import linear_model
from sklearn import tree
import numpy as np
import dataset.utility as utils
from preprocessing import preprocessing_utils as preu


class CustomersPredoctorSeparateShopCenters():

    def __init__(self, model_shopc, model_others, nsc, noth):
        self.model_shopc = model_shopc
        self.model_others = model_others
        self.models_sc = []
        self.models_o = []
        self.nsc = nsc
        self.noth = noth

    def train(self, ds):
        datassc = ds[ds['StoreType_Shopping Center'] == 1]
        datasoth = ds[ds['StoreType_Shopping Center'] == 0]
        datasc = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datassc) \
            .exclude('NumberOfSales', 'Month') \
            .build()
        print("SHOPPING CENTERS MODEL TRAINING: ")
        for i in range(self.nsc):
            print("SC :", i)
            x, y = datasc.random_sampling(1.0)
            mod = self.model_shopc()
            mod.fit(x, y)
            pr = mod.predict(datasc.xtr)
            self.models_sc.append(mod)
            print("SC ", i, " TRAIN R2: ", eval.evaluate(datasc.ytr, pr))

        dataoth = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datasoth) \
            .exclude('NumberOfSales', 'Month') \
            .build()
        print("OTHER SHOPS MODEL TRAINING: ")
        for i in range(self.noth):
            print("OTH :", i)
            x, y = dataoth.random_sampling(1.0)
            mod = self.model_others()
            mod.fit(x, y)
            pr = mod.predict(dataoth.xtr)
            self.models_o.append(mod)
            print("OTH ", i, " TRAIN R2: ", eval.evaluate(dataoth.ytr, pr))

    def predict_oth(self, x):
        preds = []
        for i in range(self.noth):
            preds.append(self.models_o[i].predict(x))
        return np.array(preds).mean(axis=0)

    def predict_sc(self, x):
        preds = []
        for i in range(self.nsc):
            preds.append(self.models_sc[i].predict(x))
        return np.array(preds).mean(axis=0)

    def test(self, ds):
        datassc = ds[ds['StoreType_Shopping Center'] == 1]
        datasoth = ds[ds['StoreType_Shopping Center'] == 0]
        datasc = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datassc) \
            .exclude('NumberOfSales', 'Month') \
            .build()
        print("SHOPPING CENTERS MODEL EVALUATION")
        preds = self.predict_sc(datasc.xts)
        print("SC TEST R2: ", eval.evaluate(datasc.yts, preds))

        dataoth = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=datasoth) \
            .exclude('NumberOfSales', 'Month') \
            .build()

        print("OTHERS MODEL EVALUATION")
        preds = self.predict_oth(dataoth.xts)
        print("OTH TEST R2: ", eval.evaluate(dataoth.yts, preds))

    def predict(self, row):
        sc = (row['StoreType_Shopping Center'] == 1).all()
        clean_row = data_manager.to_numpy(row.drop(['NumberOfSales', 'Month', 'NumberOfCustomers',
                                                   'StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover',
                                                   'Max_Sea_Level_PressurehPa',
                                                   'WindDirDegrees', 'Max_Dew_PointC', 'Mean_Sea_Level_PressurehPa',
                                                   'Min_Sea_Level_PressurehPa', 'Day'], axis=1
                                                   )).squeeze()
        clean_row = clean_row.reshape([1, -1])
        if sc:
            return self.predict_sc(clean_row).squeeze()
        else:
            return self.predict_oth(clean_row).squeeze()



def model1():
    return linear_model.Ridge(alpha=10)


def model2():
    return tree.DecisionTreeRegressor(max_depth=9)


data = data_manager.read_dataset("best_for_customers.csv")
model = CustomersPredoctorSeparateShopCenters(model1, model1, 10, 10)
model.train(data)
model.test(data)
data = utils.get_frame_in_range(data, 1, 2018, 2, 2018)
preds = []
for i in range(len(data)):
    irow = data.iloc[[i]]
    preds.append(model.predict(irow))
preds = np.array(preds)
data = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=data) \
            .exclude('NumberOfSales', 'Month') \
            .build()
print("FINAL R2: ", eval.evaluate(data.yts, preds))



