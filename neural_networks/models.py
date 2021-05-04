# First model is a dense neural network model with 5 layers
from functools import wraps

from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop


def summary(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        print(res.summary())
        return res

    return wrapper


@summary
def get_dnn_model():
    model_1 = Sequential()
    model_1.add(Dense(200, activation="relu", input_shape=(784,)))
    model_1.add(Dense(100, activation="relu"))
    model_1.add(Dense(60, activation="relu"))
    model_1.add(Dense(30, activation="relu"))
    model_1.add(Dense(10, activation="softmax"))

    # Define the optimizer and compile the model
    optimizer = optimizers.SGD(lr=0.03, clipnorm=5.)
    model_1.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model_1


@summary
def get_cnn_model():
    # Second model is a 3 layer convolutional network model with one dense layer at the end

    model_2 = Sequential()
    model_2.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, padding='Same',
                       activation='relu', input_shape=(28, 28, 1)))
    model_2.add(Conv2D(filters=8, kernel_size=(4, 4), strides=2, padding='Same',
                       activation='relu'))
    model_2.add(Conv2D(filters=12, kernel_size=(4, 4), strides=2, padding='Same',
                       activation='relu'))
    model_2.add(Flatten())
    model_2.add(Dense(200, activation="relu"))
    model_2.add(Dense(10, activation="softmax"))

    # Define the optimizer and compile the model
    optimizer = optimizers.SGD(lr=0.03, clipnorm=5.)
    model_2.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model_2


@summary
def get_hyperparam_model():
    # Fourth model with hyper parameter tuning

    model_4 = Sequential()
    model_4.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='Same',
                       activation='relu', input_shape=(28, 28, 1)))
    model_4.add(BatchNormalization())
    model_4.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='Same',
                       activation='relu'))
    model_4.add(BatchNormalization())
    model_4.add(Dropout(0.4))

    model_4.add(Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='Same',
                       activation='relu'))
    model_4.add(BatchNormalization())
    model_4.add(Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='Same',
                       activation='relu'))
    model_4.add(BatchNormalization())
    model_4.add(Dropout(0.4))

    model_4.add(Flatten())
    model_4.add(Dense(256, activation="relu"))
    model_4.add(Dropout(0.4))
    model_4.add(Dense(10, activation="softmax"))

    # Define the optimizer and compile the model
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model_4.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model_4
