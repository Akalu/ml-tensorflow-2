from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from graphic_utils.confusion_matrix import plot_confusion_matrix
from ioutils.loaders import load_data
from neural_networks.models import get_dnn_model

train_data_path = "../data/train.csv.zip"


def main():
    feature, label = load_data(train_data_path)

    # Split the dataset into train and validation set
    # Keep 10% for the validation and 90% for the training
    # Stratify is argument to keep training set evenly balanced over the labels

    feature_train, feature_test_set, label_train, label_val = train_test_split(feature, label, test_size=0.1, stratify=label)

    dnn_model = get_dnn_model()

    history = dnn_model.fit(feature_train, label_train, batch_size=100, epochs=8,
                            validation_data=(feature_test_set, label_val), verbose=1)

    # predict results
    results = dnn_model.predict(feature_test_set)

    # Convert predictions classes to one hot vectors
    prediction_classes = np.argmax(results, axis=1)
    # Convert validation observations to one hot vectors
    hits = np.argmax(label_val, axis=1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(hits, prediction_classes)
    print(confusion_mtx)
    summary = defaultdict()
    for num in range(10):
        hit = confusion_mtx[num][num]
        loss = 0
        for i in range(0,10):
            if i != num:
                loss += confusion_mtx[num][i]
                loss += confusion_mtx[i][num]
        summary[num] = f'{float(100 * hit / (hit + loss)) : .2f}'
    print(summary)


if __name__ == '__main__':
    main()
