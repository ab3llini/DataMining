import keras as k
import dataset.dataset as d
import models.keras.customers_predictor.model as m
import preprocessing.preprocessing_utils as pre_u
import evaluation.evaluation as eva
import numpy as np
import dataset.utility as utils
import pandas as p
import preprocessing.imputation as imp
import dataset.setbuilder as sb
number_of_model = 3


def evaluate(models, number_of_model, ds, y, x, number_print):
    preds = np.zeros(len(y))
    for i in range(number_of_model):
        p = models[i].predict(x, 500).squeeze()
        preds += p
    preds[preds < 0] = 0
    for i in range(min(len(preds), number_print)):
        print("PRED: ", preds[i], "   y: ", y[i])

    print("R2: ", eva.r2(ds, preds, 'NumberOfCustomers'))


if __name__ == '__main__':
    TRAIN = True
    LOAD = False
    SAVE_DS = True
    name = "lin_"
    dts = d.read_dataset("best_for_customers.csv")
    ds = sb.SetBuilder(target='NumberOfCustomers', autoexclude=True, df=dts)\
        .exclude('NumberOfSales', 'Month')\
        .build()
    models = []
    x, y = ds.xtr, ds.ytr
    real_y = np.array(y)
    x_test, y_test = ds.xts, ds.yts
    for i in range(number_of_model):
        if not LOAD:
            models.append(m.single_relu(x.shape[1], i == 0))
        else:
            models.append(k.models.load_model("mod" + name + str(i) + ".h5"))
        opt = k.optimizers.adam(lr=2e-4)
        models[i].compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        models[i].summary()
        if TRAIN:
            models[i].fit(x=x, y=y, batch_size=20000, epochs=80, verbose=2, validation_data=(x_test, y_test))
            models[i].save("mod" + name + str(i) + ".h5")
            dy = models[i].predict(x, 500)
            print(dy.shape, y.shape)
            y = y.squeeze() - dy.squeeze()
        print(y)
        y.reshape([len(y), 1])

    print("################################################")
    print("EVALUATION ON TEST:")
    evaluate(models, number_of_model, utils.get_frame_in_range(dts, 1, 2018, 2, 2018), y_test, x_test, 1000)

    print("################################################")
    print("EVALUATION ON TRAIN:")
    evaluate(models, number_of_model, utils.get_frame_in_range(dts, 2, 2016, 12, 2017), real_y, x, 0)