from sys import argv
from pathlib import Path
from keras.models import load_model
from keras.utils import plot_model

path = Path(argv[1])
model = load_model(str(path), compile=False)
plot_model(model, to_file='{}.png'.format(path.stem))
