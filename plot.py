from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(argv[1])

keys = ['loss', 'val_loss']
ax = df[keys].plot(kind='line')
ax.set_xlabel('epoch')
ax.set_ylabel('loss(cross entropy)')
plt.savefig('loss.png')

keys = ['binary_accuracy', 'val_binary_accuracy']
ax = df[keys].plot(kind='line')
ax.set_xlabel('epoch')
ax.set_ylabel('loss(cross entropy)')
plt.savefig('acc.png')
