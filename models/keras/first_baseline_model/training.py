import keras as k
import dataset.dataset as d
import models.keras.first_baseline_model.model as m
import preprocessing.preprocessing_utils as pre_u

if __name__ == '__main__':
    ds = d.read_imputed_onehot_dataset()
    ds = pre_u.eliminate_zeros(ds)
    y = d.to_numpy(ds[['NumberOfSales']])
    x = d.to_numpy(ds.drop(['NumberOfSales', 'StoreID', 'Date'], axis=1))
    print(x.shape, y.shape)
    model = m.nonsequentialNN(x.shape[1])
    opt = k.optimizers.adam(lr=1e-4)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    model.summary()
    model.fit(x=x, y=y, batch_size=500, epochs=50, verbose=1)
