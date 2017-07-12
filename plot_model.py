from sys import argv
from pathlib import Path
from keras.models import load_model
from keras.utils import plot_model

model = load_model(argv[1], compile=False)
plot_model(model, to_file='{}.png'.format(argv[2]), show_shapes=True)
