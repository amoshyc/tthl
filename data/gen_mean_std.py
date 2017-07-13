import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image

def gen_mean_std():
    video_dir = Path('~/dataset/video00').expanduser()
    frame_dir = video_dir / 'frames/'
    img_paths = sorted(frame_dir.iterdir())

    N = len(img_paths)
    x_mean = np.zeros((224, 224, 3), dtype=np.float32)
    x_mean_sq = np.zeros((224, 224, 3), dtype=np.float32)

    for path in tqdm(img_paths):
        pil = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(pil)

        x_mean += x / N
        x_mean_sq += (x ** 2) / N
    
    x_std = np.sqrt(x_mean_sq - x_mean ** 2)

    np.save('mean.npy', x_mean)
    np.save('std.npy', x_std)

if __name__ == '__main__':
    gen_mean_std()