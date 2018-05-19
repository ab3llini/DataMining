import keras as k
import dataset.dataset as d
import models.keras.first_baseline_model.model as m
import preprocessing.preprocessing_utils as pre_u
import models.keras.evaluation as eval
import numpy as np
number_of_model = 3


def prepare_ds(ds):
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.add_avg_per_shop(ds)
    ds = pre_u.add_std_per_shop(ds)
    ds = pre_u.add_max_per_shop(ds)
    ds = pre_u.add_min_per_shop(ds)
    return ds


if __name__ == '__main__':
    TRAIN = True
    LOAD = False
    ds = d.read_imputed_onehot_dataset()
    ds = prepare_ds(ds)
    y = d.to_numpy(ds[['NumberOfSales']])
    real_y = np.array(y)
    dy = np.zeros(y.shape)
    x = d.to_numpy(ds.drop(['NumberOfSales', 'StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover',
                            'Max_Sea_Level_PressurehPa', 'WindDirDegrees', 'Max_Dew_PointC'], axis=1))
    models = []
    for i in range(number_of_model):
        if not LOAD:
            models.append(m.nonsequentialNN(x.shape[1], i == 0))
        else:
            models.append(k.models.load_model("mod" + str(i) + ".h5"))
        opt = k.optimizers.adam(lr=1e-4)
        models[i].compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        models[i].summary()
        if TRAIN:
            models[i].fit(x=x, y=y, batch_size=500, epochs=15, verbose=1)
            models[i].save("mod" + str(i) + ".h5")
            dy = models[i].predict(x, 500)
            print(dy.shape, y.shape)
        y = y.squeeze() - dy.squeeze()
        print(y)
        y.reshape([len(y), 1])

    preds = np.zeros(y.shape)
    for i in range(number_of_model):
        preds += models[i].predict(x, 500).squeeze()
    preds[preds < 0] = 0
    number = 1000

    for i in range(min(len(preds), number)):
        print("PRED: ", preds[i], "   x: ", real_y[i])

    print("R2: ", eval.r2(ds, preds, 'NumberOfSales'))
