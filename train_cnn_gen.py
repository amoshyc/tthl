import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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

def generator(xs, ys, batch_size):
    x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    while True:
        for i, img_path in enumerate(xs):
            idx = i % batch_size
            pil = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(pil)
            x_batch[idx] = img
            y_batch[idx] = ys[i]

            if idx == batch_size - 1:
                yield (x_batch, y_batch)


def get_data(n_samples, batch_size):
    video_dir = Path('~/dataset/video00').expanduser()
    frame_dir = video_dir / 'frames/'
    info_path = video_dir / 'info.json'

    x_all = sorted(frame_dir.iterdir())
    y_all = json.load(info_path.open())['label']

    indices = np.random.permutation(len(x_all))[:n_samples]
    x_use = [x_all[i] for i in indices]
    y_use = [y_all[i] for i in indices]

    pivot = n_samples * 4 // 5
    x_train, x_val = x_use[:pivot], x_use[pivot:]
    y_train, y_val = y_use[:pivot], y_use[pivot:]

    train_gen = generator(x_train, y_train, batch_size)
    val_gen = generator(x_val, y_val, batch_size)

    return (train_gen, val_gen)


def main():
    with tf.device('/gpu:3'):
        model = get_model()

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])
    model.summary()

    n_samples = 27000
    n_train = n_samples * 4 // 5
    n_val = n_samples // 5
    batch_size = 40
    train_gen, val_gen = get_data(n_samples, batch_size)

    arg = {
        'generator': train_gen,
        'steps_per_epoch': n_train // batch_size,
        'epochs': 30,
        'validation_data': val_gen, 
        'validation_steps': n_val // batch_size,
        'callbacks': [
            CSVLogger('cnn.log'),
            ModelCheckpoint(filepath="./cnn_epoch{epoch:02d}_{val_binary_accuracy:.3f}.h5")
        ]
    } # yapf: disable

    model.fit_generator(**arg)


if __name__ == '__main__':
    main()