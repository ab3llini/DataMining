import keras.layers as klayers
import keras as k


def simpleNN(num):
    input = klayers.Input(shape=(num,))
    x = klayers.Dense(units=300, activation='linear', use_bias=True)(input)
    x = klayers.Dense(units=1, activation='relu')(x)
    return k.Model(inputs=input, outputs=x)


def nonsequentialNN(num):
    inputs = klayers.Input(shape=(num,))
    y = klayers.Dense(units=400, activation='relu', use_bias=True)(inputs)
    x = klayers.Dense(units=400, activation='linear', use_bias=True)(inputs)
    x = klayers.Dense(units=1, activation='relu')(klayers.concatenate([x, y]))
    return k.Model(inputs=inputs, outputs=x)
