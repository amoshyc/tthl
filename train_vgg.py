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
from keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.vgg16 import VGG16


class MyLogger(Callback):
    def __init__(self, prefix=None):
        self.train_loss = []
        self.val_loss = []
        self.loss_path = Path('{}_loss.json'.format(prefix))
        self.acc_path = Path('{}_acc.json'.format(prefix))

        # self.loss_path = self.loss_path.resolve()
        # self.acc_path = self.acc_path.resolve()
        # self.loss_path.parent.mkdir(parents=True, exist_ok=True)
        # self.acc_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('acc'))

        df_loss = pd.DataFrame()
        df_loss['train_loss'] = self.train_loss
        df_loss.to_json(str(self.loss_path), orient='split')

        df_loss = pd.DataFrame()
        df_loss['train_acc'] = self.train_acc
        df_loss.to_json(str(self.acc_path), orient='split')


def get_model():
    vgg = VGG16(weights='imagenet', include_top=False, pooling='max')
    x = vgg.output
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='softmax')(x)

    return Model(inputs=vgg.input, outputs=x)


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

    opt = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    model.summary()

    n_use = 27000
    batch_size = 40

    arg = {
        'generator': generator(n_use, batch_size),
        'steps_per_epoch': n_use // batch_size,
        'epochs': 30,
        'callbacks': [
            MyLogger(prefix='vgg'),
            ModelCheckpoint(filepath="./vgg_epoch{epoch:02d}_{loss:.3f}.h5")
        ]
    } # yapf: disable

    model.fit_generator(**arg)


if __name__ == '__main__':
    main()