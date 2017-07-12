from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

df = pd.read_csv(argv[1])

ax = df.plot(kind='line', x='loss', y='val_loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss(cross entropy)')
plt.savefig('loss.png')

ax = df.plot(kind=line, x='train_binary_accuracy', y='val_binary_accuracy')
ax.set_xlabel('epoch')
ax.set_ylabel('loss(cross entropy)')
plt.savefig('acc.png')
