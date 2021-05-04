# Generate 22 million more images by randomly rotating, scaling, and shifting 42,000 (-10% validation set) images
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def get_images(feature_train):
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        featurewise_center=False,  # do not set input mean to 0 over the dataset
        samplewise_center=False,  # do not set each sample mean to 0
        featurewise_std_normalization=False,  # no divide inputs by std of the dataset
        samplewise_std_normalization=False,  # no divide each input by its std
        zca_whitening=False,  # No ZCA whitening
        horizontal_flip=False,  # no horizontal flip images
        vertical_flip=False)  # no vertical flip images, no 6 and 9 mismatches :-)

    datagen.fit(feature_train)
    return datagen