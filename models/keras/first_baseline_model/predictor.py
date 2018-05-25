import keras as k
import numpy as np
import os


class SalesPredictor:

    def __init__(self, n, name=""):
        self.models = []
        self.n = n
        _dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(n):
            self.models.append(k.models.load_model(os.path.join(_dir, "mod" + name + str(i) + ".h5")))

    def predict(self, x):
        preds = np.zeros(len(x))
        for i in range(self.n):
            p = self.models[i].predict(x).squeeze()
            preds += p
        preds[preds < 0] = 0
        return preds

    def get_weights(self, modnum):
        return self.models[modnum].get_weights()


if __name__ == '__main__':
    pred = SalesPredictor(3, "test")
    wl = pred.get_weights(0)[0]
    print(wl)
    print(np.max(wl))
    print(np.min(wl))
