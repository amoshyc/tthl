from pprint import pprint
from pathlib import Path
from sys import argv
from keras.models import load_model
from generator import generator

model = load_model(argv[1])
x_val = np.load('x_val_std.npy')[:10]
y_val = np.load('y_val.npy')[:10]

y_pred = model.predict(x_val, y_val, batch_size=10)
pprint(y_pred)
print('*' * 50)
pprint(y_val)
