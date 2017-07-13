from pprint import pprint
from pathlib import Path
from sys import argv
import numpy as np
from keras.models import load_model

model = load_model(argv[1])

res = model.evaluate(x_val, y_val)
print(res)