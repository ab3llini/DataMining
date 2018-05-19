import keras as k
import dataset.dataset as d
import models.keras.first_baseline_model.model as m
import preprocessing.preprocessing_utils as pre_u
import models.keras.evaluation as eval
import numpy as np
import dataset.utility as utils
number_of_model = 3


def prepare_ds(ds):
    ds = pre_u.eliminate_IsOpen_zeros(ds)
    ds = pre_u.add_avg_per_shop(ds)
    ds = pre_u.add_std_per_shop(ds)
    ds = pre_u.add_max_per_shop(ds)
    ds = pre_u.add_min_per_shop(ds)
    return ds


def prepare_out(ds):
    y = d.to_numpy(ds[['NumberOfSales']])
    return y


def drop_useless(ds):
    x = d.to_numpy(ds.drop(['NumberOfSales', 'StoreID', 'Date', 'IsOpen', 'Region', 'CloudCover',
                            'Max_Sea_Level_PressurehPa', 'WindDirDegrees', 'Max_Dew_PointC'], axis=1))
    return x


if __name__ == '__main__':
    TRAIN = False
    LOAD = True
    ds = d.read_imputed_onehot_dataset()
    ds = prepare_ds(ds)
    ds_train = utils.get_frame_in_range(ds, 3, 2016, 12, 2017)
    ds_test = utils.get_frame_in_range(ds, 1, 2018, 2, 2018)
    y = prepare_out(ds_train)
    real_y = np.array(y)
    dy = np.zeros(y.shape)
    x = drop_useless(ds_train)
    y_test = prepare_out(ds_test)
    x_test = drop_useless(ds_test)
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

    preds = np.zeros(len(y_test))
    for i in range(number_of_model):
        p = models[i].predict(x_test, 500).squeeze()
        preds += p
    preds[preds < 0] = 0
    number = 1000

    for i in range(min(len(preds), number)):
        print("PRED: ", preds[i], "   y_test: ", y_test[i])

    print("R2: ", eval.r2(ds_test, preds, 'NumberOfSales'))
