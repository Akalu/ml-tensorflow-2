import pandas as pd
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding


def load_data(train_ds_path, compr='zip'):
    # Load the training data
    dataset = pd.read_csv(train_ds_path, compression=compr)

    # A label is the thing we're predicting
    label = dataset["label"]

    # A feature is an input variable, in this case a 28 by 28 pixels image
    # Drop 'label' column
    feature = dataset.drop(labels=["label"], axis=1)

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    label = to_categorical(label, num_classes=10)

    # Normalize between 0 and 1 the data (The pixel-value is an integer between 0 and 255)
    feature = feature / 255.0

    del dataset

    return feature, label
