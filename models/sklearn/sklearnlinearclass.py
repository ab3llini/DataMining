from sklearn import linear_model
import numpy as np


class LinearSklearn:
    def __init__(self, nmodels):
        self.models = []
        self.n = nmodels
        for _ in range(nmodels):
            self.models.append(linear_model.LinearRegression())

    def train(self, x, y):
        y = y.squeeze()
        to_pred = y
        weights = np.ones(shape=y.shape[0], dtype=np.float32)
        for i in range(self.n):
            print("Training model: " + str(i))
            self.models.append(linear_model.LinearRegression())
            self.models[i].fit(x, to_pred, sample_weight=weights)
            preds = self.models[i].predict(x)
            print(preds)
            to_pred = to_pred - preds

    def predict(self, x):
        ypred = np.zeros(shape=[x.shape[0]], dtype=np.float32)

        for i in range(self.n):
            ypred += self.models[i].predict(x)
        return ypred