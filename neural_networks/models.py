# First model is a dense neural network model with 5 layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras import optimizers


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

    print(model_1.summary())
    return model_1
