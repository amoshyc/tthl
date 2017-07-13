import json
from pathlib import Path

from data import gen_img_label

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


def get_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(4, kernel_size=5, strides=3, activation='relu'))
    model.add(Conv2D(8, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(12, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(30))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


def main():
    # with tf.device('/gpu:3'):
        # model = get_model()
    model = get_model()

    model_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['binary_accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    n_train = 27000
    n_val = 1000
    batch_size = 40

    dataset = Path('~/dataset/').expanduser().resolve()
    train_dirs = [(dataset / 'video00/')]
    val_dirs = [(dataset / 'video01')]
    train_gen = gen_img_label(train_dirs, n_train, batch_size)
    val_gen = gen_img_label(val_dirs, n_val, batch_size)

    fit_arg = {
        'generator': train_gen,
        'steps_per_epoch': n_train // batch_size,
        'epochs': 30,
        'validation_data': val_gen,
        'validation_steps': n_val // batch_size,
        'callbacks': [
            CSVLogger('cnn.log'),
            ModelCheckpoint(filepath="/tmp/cnn_epoch{epoch:02d}_{val_binary_accuracy:.3f}.h5")
        ]
    } # yapf: disable

    model.fit_generator(**fit_arg)


if __name__ == '__main__':
    main()