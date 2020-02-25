import matplotlib.pyplot as plt

import plot_utils as pu
import numpy as np
from PIL import Image
from matplotlib.colors import NoNorm

pu.figure_setup()

fig_size = pu.get_fig_size(10, 8)
print(fig_size)
fig = plt.figure(figsize=fig_size)

ax = fig.add_subplot(121)

img = Image.open('../output/X_REAL/0010.png').convert('L')
arr = np.asarray(img)
plt.imshow(img,cmap='gray',norm=NoNorm())
ax.set_title('Original')

ax = fig.add_subplot(122)
img = Image.open('../output/Y/0010.png').convert('L')
arr = np.asarray(img)
plt.imshow(img,cmap='gray',norm=NoNorm())
ax.set_title('Translated')

pu.save_fig(fig, 'img2img', 'pdf')