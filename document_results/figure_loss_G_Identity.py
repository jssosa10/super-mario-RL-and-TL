import matplotlib.pyplot as plt
import plot_utils as pu
import pandas as pd

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

data = pd.read_csv("run-Dec08_17-57-30_juansosa-Lenovo-Legion-Y7000P-1060-tag-Loss_G_identity.csv")
data = data.fillna(value=0)
data.head()

pu.figure_setup()

fig_size = pu.get_fig_size(10, 8)
print(fig_size)
fig = plt.figure(figsize=fig_size)
#fig = plt.figure()
ax = fig.add_subplot(111)
dt = data['Value'].to_numpy()

smt = smooth(dt,0.99)

ax.plot(range(len(data['Value'])), data['Value'], c='orange', linewidth=0.5, alpha=0.3)
ax.plot(range(len(data['Value'])), smt, c='orange', linewidth=0.5)
ax.set_xlabel('Steps')
ax.set_ylabel('Loss G Identity')

#ax.set_axisbelow(True)

#plt.grid()
#plt.tight_layout()
pu.save_fig(fig, 'loss_G_Iden', 'pdf')
