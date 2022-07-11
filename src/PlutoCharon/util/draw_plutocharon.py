# Utility to draw circles around Pluto and Charon given their locations

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from Image import Image
from res import config

x_p = 8.956927599056726
y_p = 10.915912980619341
x_c = 11.374072984630411
y_c = 5.8357916026729475


path = (
    f'{config.output_folder}/{config.date}/{config.date}{config.index}_PC_subimage.fits'
)
img = Image(path)
plt.figure(figsize=(8, 8))
display_image = img.data.astype(float)

# min_clip = 4000 4-25-2021
min_clip = 490
vmax = 580

display_image[display_image < min_clip] = (
    min_clip + 1
)  # will remove the 'static' of white dots
plt.imshow(display_image, norm=LogNorm(vmin=min_clip, vmax=vmax), cmap='Greys_r')

plt.scatter([x_c, x_p], [y_c, y_p], facecolors='none', edgecolors=['b', 'r'])
ax = plt.gca()
ax.invert_yaxis()
plt.show()
