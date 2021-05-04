import numpy as np
import matplotlib.pyplot as plt


def draw_symbols(feature, label, nrows=2, ncols=3):
    """This shows 6 random images with their labels"""
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            rand_example = np.random.choice(feature.index)
            ax[row, col].imshow(feature.loc[rand_example].values.reshape((28, 28)), cmap='gray_r')
            ax[row, col].set_title("Label: {}".format(label.loc[rand_example]))
