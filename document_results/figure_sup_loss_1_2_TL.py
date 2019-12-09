import matplotlib.pyplot as plt
import plot_utils as pu
import pandas as pd

data = pd.read_csv("run-Dec02_23-30-08_juansosa-Lenovo-Legion-Y7000P-1060-tag-Loss_supervised.csv")
data = data.fillna(value=0)
data.head()

pu.figure_setup()

fig_size = pu.get_fig_size(10, 8)
print(fig_size)
fig = plt.figure(figsize=fig_size)
#fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(data['Step'], data['Value'], c='g', linewidth=0.5)

ax.set_xlabel('Steps')
ax.set_ylabel('Supervised Loss')

#ax.set_axisbelow(True)

#plt.grid()
#plt.tight_layout()
pu.save_fig(fig, '1-2lossSupTL', 'pdf')
