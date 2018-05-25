import numpy as np


class LinearSklearn:
    def __init__(self, nmodels, mod):
        self.models = []
        self.n = nmodels
        for _ in range(nmodels):
            self.models.append(mod())

    def train(self, x, y):
        y = y.squeeze()
        to_pred = y
        weights = np.ones(shape=y.shape[0], dtype=np.float32)
        for i in range(self.n):
            print("Training model: " + str(i))
            self.models[i].fit(x, to_pred)
            preds = self.models[i].predict(x)
            to_pred = to_pred - preds

    def predict(self, x):
        ypred = np.zeros(shape=[x.shape[0]], dtype=np.float32)
        for i in range(self.n):
            ypred += self.models[i].predict(x)
        return ypred


    def print_weights(self):
        for i in range(self.n):
            print(self.models[i].coef_)
