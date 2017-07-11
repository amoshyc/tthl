from random import randint
from pathlib import Path
from collections import deque
import numpy as np
from keras.preprocessing import image


def sliding_window(self, frame_paths, length, overlap):
    n, k = length, overlap
    window = np.zeros((n, 224, 224, 3), dtype=np.float32)

    # n = 9, k = 4
    # xxxxxoooo
    #      ooooxxxxx

    window = deque()
    for path in frame_paths[:n]:
        img = load_img(path, target_size=(224, 224))
        window.append(img)
    yield np.array(window, dtype=float) / 255.0

    for s in range(n - k, n, n - k):
        for i in range(n - k):
            window.popleft()
            img = load_img(frame_paths[s + i], target_size=(224, 224))
            window.append(img)
        yield np.array(window, dtype=float) / 255.0
        
