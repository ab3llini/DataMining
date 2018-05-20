import keras as k
import dataset.dataset as d
import models.keras.customers_predictor.model as m
import preprocessing.preprocessing_utils as pre_u
import models.keras.evaluation as eva
import numpy as np
import dataset.utility as utils
import pandas as p
import preprocessing.imputation as imp
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