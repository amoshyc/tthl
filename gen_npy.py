import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image


def gen_npy(n_use):
    video_dir = Path('~/dataset/video00').expanduser()
    frame_dir = video_dir / 'frames/'
    info_path = video_dir / 'info.json'

    img_paths = sorted(frame_dir.iterdir())
    info = json.load(info_path.open())

    # indices = np.random.permutation(len(img_paths))[:n_use]
    indices = range(n_use)
    img_uses = [img_paths[i] for i in indices]
    label_uses = [info['label'][i] for i in indices]

    print('Loading...')
    x = np.zeros((n_use, 224, 224, 3), dtype=float)
    y = np.array(label_uses, dtype=int)
    for i, path in enumerate(tqdm(img_uses)):
        pil = image.load_img(path, target_size=(224, 224))
        x[i] = image.img_to_array(pil)

    pivot = n_use * 4 // 5
    x_train, x_val = x[:pivot], x[pivot:]
    y_train, y_val = y[:pivot], y[pivot:]

    print('Standardization...')
    std = image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True)
    std.fit(x_train)
    x_train_std = std.standardize(x_train)
    x_val_std = std.standardize(x_val)

    print('Saving...')
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)
    np.save('x_train_std.npy', x_train_std)
    np.save('x_val_std.npy', x_val_std)


if __name__ == '__main__':
    gen_npy()