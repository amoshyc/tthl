import json
from pathlib import Path

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
from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.vgg16 import VGG16


class MyLogger(Callback):
    def __init__(self, loss_path=None):
        self.train_loss = []
        self.val_loss = []
        self.loss_path = loss_path or './loss.json'

    def on_epoch_end(self, epoch, logs):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        df_loss = pd.DataFrame()
        df_loss['train_loss'] = self.train_loss
        df_loss['val_loss'] = self.val_loss
        df_loss.to_json(self.loss_path, orient='split')


def get_model():
    img = Input(shape=(224, 224, 3))

    

    return Model(inputs=img, outputs=x)


def generator(n_use, batch_size):
    video_dir = Path('~/dataset/video00').expanduser()
    frame_dir = video_dir / 'frames/'
    info_path = video_dir / 'info.json'

    img_paths = sorted(frame_dir.iterdir())
    info = json.load(info_path.open())

    x_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.uint8)

    indices = np.random.permutation(len(img_paths))[:n_use]
    img_uses = [img_paths[i] for i in indices]
    label_uses = [info['label'][i] for i in indices]

    while True:
        for i, img_path in enumerate(img_uses):
            idx = i % batch_size
            x_batch[idx] = image.load_img(img_path, target_size=(224, 224))
            y_batch[idx] = label_uses[i]

            if idx == batch_size - 1:
                x_batch /= 255
                yield (x_batch, y_batch)


def main():
    with tf.device('/gpu:3'):
        model = get_model()

    # model = get_model()

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
    model.summary()

    n_use = 27000
    batch_size = 40

    arg = {
        'generator': generator(n_use, batch_size),
        'steps_per_epoch': n_use // batch_size,
        'epochs': 30,
        'callbacks': [
            # MyLogger(loss_path='./vgg_loss.json'),
            ModelCheckpoint(filepath="./vgg_epoch{epoch:02d}_{train_loss:.3f}.h5")
        ]
    } # yapf: disable

    model.fit_generator(**arg)


if __name__ == '__main__':
    main()