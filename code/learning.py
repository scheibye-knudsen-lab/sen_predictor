from sampler import SampleManager
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, Flatten, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow import keras
import tensorflow as tf

import numpy as np
import os
import math
from sklearn.utils import class_weight
import prep
import learning_xception


EPOCH_COUNT = 10

LEARNING_RATE = 1e-4


BATCH_SIZE = 16

STANDARDIZED = 1

AUG_CYCLES = 1
AUG_CYCLES_VAL = 1

BALANCE_SAMPLES = True   # auto
# BALANCE_SAMPLES = {0: 1, 1: 4}   # custom

MAX_CLASS_WEIGHT_RATIO = 40

# shared seed for fit and flow to sync generation
FLOW_SEED = None


class AccuracyCallback(Callback):

    def __init__(self, data, classes, predictor):
        self.data = data
        self.classes = classes
        self.predictor = predictor

    def on_epoch_end(self, epoch, logs=None):
        xs, ys = self.data

        # ys = prep.hot_ys(y_labels, sample_classes)

        probys = self.predictor(xs)
        pys = prep.onehot(probys)
        print()
        prep.summarize(ys, pys, self.classes)
        print()


class LearningAgent():

    def __init__(self, model_filename, img_shape, do_regression=False, target_classes=None):

        self.target_classes = target_classes
        self.img_shape = img_shape
        self.model_filename = model_filename
        self.do_regression = do_regression

        self.model = None
        self._ready = False

    def _init(self, sample_num=1):
        if self._ready:
            return

        self._ready = True

        self._init_model(sample_num)

        print(self.model.summary())

    def _init_model(self, sample_num=1):
        self.model = learning_xception._make_model(
            self.img_shape, self.target_classes, regression=self.do_regression)

    def _create_aug_gen(self):
        return ImageDataGenerator(
            data_format='channels_last',
            fill_mode="constant",
            cval=0,

            featurewise_center=False,
            featurewise_std_normalization=False,

            samplewise_center=False,
            samplewise_std_normalization=False,

            # rescale=1.0 / 255.0,  # so 8b is 0 to 1

            # zca_epsilon
            # zca_whitening
            # channel_shift_range
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # shear_range=0.1,

            # zoom_range=[0.8, 1.0],
            # brightness_range=[0.7, 1.3],
            # zoom_range=[0.8, 1.2],

            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=180,
        )

    def get_model_fn(self):
        return self.model_filename

    def load_weights(self):
        model_fn = self.get_model_fn()
        print("Loading model weights: ", model_fn)
        if os.path.exists(model_fn):
            self.model.load_weights(model_fn)
            return True
        else:
            print("*** I am Weightless")
            return False

    def train(self, train_x, train_y, val_x=None, val_y=None):
        self._init(sample_num=len(train_x))

        self.do_train(train_x, train_y, val_x, val_y)

    def do_train(self, train_x, train_y, val_x=None, val_y=None):

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        if STANDARDIZED:
            train_x = prep.standardize_colors(train_x)

        if val_x is not None:
            val_x = np.asarray(val_x)
            val_y = np.asarray(val_y)
            if STANDARDIZED:
                val_x = prep.standardize_colors(val_x)

        val_x, val_y = self.balance(val_x, val_y)

        self.do_train_model(train_x, train_y, val_x, val_y)

    def do_train_model(self, train_x, train_y, val_x, val_y):
        self.load_weights()

        if self.do_regression:
            monitor = "mean_squared_error"
            monitor_mode = "min"
        elif len(self.target_classes) > 1:
            monitor = "categorical_accuracy"
            # monitor = # "precision" #metrics.Precision()
            monitor_mode = "max"
        else:
            monitor = "accuracy"
            monitor_mode = "max"

        callbacks = []
        if 1:
            EarlyStopping(monitor="val_loss", patience=3, verbose=1),
            cb = ReduceLROnPlateau(factor=0.1, patience=3, monitor=monitor,
                                   min_lr=1e-7, verbose=1)
            callbacks.append(cb)

        model_fn = self.get_model_fn()
        checkpoint_best_model = ModelCheckpoint(model_fn, monitor=monitor, mode=monitor_mode,
                                                verbose=1, save_best_only=True, save_weights_only=True)
        callbacks.append(checkpoint_best_model)

        def model_predict(xs):
            return self.do_predict(xs)

        acc_cb = AccuracyCallback(
            [val_x, val_y], self.target_classes, model_predict)
        callbacks.append(acc_cb)

        opti = optimizers.Adam(lr=LEARNING_RATE)

        if len(self.target_classes) > 1:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=opti, metrics=[monitor])
        elif self.do_regression:
            self.model.compile(loss='mean_squared_error',
                               optimizer=opti, metrics=['mean_absolute_error', monitor])
        else:
            self.model.compile(loss='binary_crossentropy',
                               optimizer=opti, metrics=[monitor])

        if self.do_regression:
            class_weights = None
        else:
            class_weights = self.prepare_class_weights(train_y)

        print(train_y[0:10])
        print("Balancing weights:", class_weights)
        if max(class_weights.values()) / min(class_weights.values()) > MAX_CLASS_WEIGHT_RATIO:
            print("class_weight ratio exceeded MAX, adjusting...")
            mk = min(class_weights)
            class_weights[mk] = 1
            mk = max(class_weights)
            class_weights[mk] = MAX_CLASS_WEIGHT_RATIO
            print("Balancing weights:", class_weights)

        if AUG_CYCLES > 0:

            train_steps = AUG_CYCLES * len(train_x) // BATCH_SIZE
            val_steps = AUG_CYCLES_VAL * len(val_x) // BATCH_SIZE

            xgen = self._create_aug_gen()

            xflow = xgen.flow(train_x, train_y, shuffle=True,
                              batch_size=BATCH_SIZE, seed=FLOW_SEED)

            if val_x is not None:
                val_flow = xgen.flow(val_x, val_y, shuffle=True)

            results = self.model.fit(
                xflow,
                steps_per_epoch=train_steps,
                epochs=EPOCH_COUNT,
                workers=9,
                verbose=1,
                shuffle=True,
                callbacks=callbacks,
                validation_data=val_flow,
                validation_steps=val_steps,
                class_weight=class_weights)

        else:
            print("*** Augmentation off")
            val_data = None
            if val_x is not None:
                val_data = (val_x, val_y)

            results = self.model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, callbacks=callbacks,
                                     validation_data=val_data, class_weight=class_weights)
        return results

    def prepare_class_weights(self, train_y):
        total = np.sum(train_y)
        bin_cnt = np.sum(train_y, axis=0)
        class_weights = None
        if BALANCE_SAMPLES == True:
            # prevent / 0
            bin_cnt = np.where(bin_cnt == 0, 1e9, bin_cnt)
            class_weights = total / (len(bin_cnt) * bin_cnt)
            if self.do_regression:
                class_weights = {k: class_weights[idx]
                                 for idx, k in enumerate(self.target_classes)}

            else:
                class_weights = {idx: k for idx, k in enumerate(class_weights)}
        elif BALANCE_SAMPLES:
            class_weights = BALANCE_SAMPLES

        return class_weights

    # always loads best weights automatically
    def predict(self, xs):
        self._init()

        xs = np.asarray(xs)
        if STANDARDIZED:
            xs = prep.standardize_colors(xs)

        py_batch = []

        has_weights = self.load_weights()
        if not has_weights:
            raise(Exception("*** Predicting without weights.  All stop!!!"))

        pys = self.do_predict(xs)

        return pys

    def do_predict(self, xs):

        proby = self.model.predict(xs)

        return proby
