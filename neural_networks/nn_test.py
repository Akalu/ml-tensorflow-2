import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from graphic_utils.confusion_matrix import get_summary
from ioutils.loaders import load_data
from neural_networks.image_generator import get_images
from neural_networks.models import get_dnn_model, get_cnn_model, get_hyperparam_model

train_data_path = "../data/train.csv.zip"


def evaluate_model(model, feature_train, feature_test_set, label_train, label_val, history):
    print(history.history)
    # predict results
    results = model.predict(feature_test_set)

    # Convert predictions classes to one hot vectors
    prediction_classes = np.argmax(results, axis=1)

    # Convert validation observations to one hot vectors
    hits = np.argmax(label_val, axis=1)

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(hits, prediction_classes)
    print(confusion_mtx)

    print(get_summary(confusion_mtx))


def main():
    feature, label = load_data(train_data_path)

    # Split the dataset into train and validation set
    # Keep 10% for the validation and 90% for the training
    # Stratify is argument to keep training set evenly balanced over the labels
    feature_train, feature_test_set, label_train, label_val = train_test_split(feature, label, test_size=0.1,
                                                                               stratify=label)
    ## DNN
    dnn_model = get_dnn_model()
    history = dnn_model.fit(feature_train, label_train, batch_size=100, epochs=8,
                            validation_data=(feature_test_set, label_val), verbose=1)

    evaluate_model(dnn_model, feature_train, feature_test_set, label_train, label_val, history)

    ## CNN
    feature = feature.values.reshape(-1, 28, 28, 1)

    feature_train, feature_test_set, label_train, label_val = train_test_split(feature, label, test_size=0.1,
                                                                               stratify=label)

    cnn_model = get_cnn_model()
    history = cnn_model.fit(feature_train, label_train, batch_size=100, epochs=8,
                            validation_data=(feature_test_set, label_val), verbose=1)

    evaluate_model(cnn_model, feature_train, feature_test_set, label_train, label_val, history)

    # Hyperparameters optimization (tuning)

    hpt_model = get_hyperparam_model()
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5,
                                                min_lr=0.00001)
    feature_train, feature_test_set, label_train, label_val = train_test_split(feature, label, test_size=0.1,
                                                                               stratify=label)
    data_generator = get_images(feature_train)

    history = hpt_model.fit(data_generator.flow(feature_train, label_train, batch_size=100),
                            epochs=3, validation_data=(feature_test_set, label_val),
                            verbose=2, callbacks=[learning_rate_reduction])

    evaluate_model(hpt_model, feature_train, feature_test_set, label_train, label_val, history)


if __name__ == '__main__':
    main()
