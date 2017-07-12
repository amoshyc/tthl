import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

df_loss = pd.read_json('./cnn_loss.json', orient='split')
ax = df_loss.plot(kind='line')
ax.set_xlabel('epoch')
ax.set_ylabel('loss(cross entropy)')
plt.savefig('loss.png')

df_acc = pd.read_json('./cnn_acc.json', orient='split')
ax = df_acc.plot(kind='line')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
plt.savefig('acc.png')