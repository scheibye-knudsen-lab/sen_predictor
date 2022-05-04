import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten


def _make_model(img_shape, classes, regression):
    xception_model = Xception(
        input_shape=img_shape, weights='imagenet', include_top=False, pooling='avg')

    model = Sequential()
    model.add(xception_model)
    model.add(Flatten())

    if regression:
        model.add(Dense(1, activation='linear'))

    else:
        model.add(
            Dense(len(classes), activation='softmax'))

    return model
