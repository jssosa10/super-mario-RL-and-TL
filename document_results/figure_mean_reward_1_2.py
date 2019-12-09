import matplotlib.pyplot as plt
import plot_utils as pu
import pandas as pd

data = pd.read_csv("run-Nov11_12-54-53_juansosa-Lenovo-Legion-Y7000P-1060-tag-Mean_Reward.csv")
data = data.fillna(value=0)

data2 = pd.read_csv("run-Nov21_06-48-58_juansosa-Lenovo-Legion-Y7000P-1060-tag-Mean_Reward.csv")
data2 = data2.fillna(value=0)

pu.figure_setup()

fig_size = pu.get_fig_size(10, 8)
print(fig_size)
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111)

ax.plot(data2['Step'], data2['Value'], c='b', linewidth=0.5, label='warm start')
ax.plot(data['Step'], data['Value'], c='g', linewidth=0.5, label='cold start')




ax.set_xlabel('Steps')
ax.set_ylabel('Mean Reward')
plt.legend()

#ax.set_axisbelow(True)

#plt.grid()
#plt.tight_layout()
pu.save_fig(fig, '1-2rew', 'pdf')
