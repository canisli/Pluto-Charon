# Utility to draw circles around Pluto and Charon given their locations

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from Image import Image
import config

x_p = 9.52835602
y_p = 9.44556182
x_c = 9.52135037
y_c = 9.43923984

path = config.data_folder + "/4-25-2021/plutocharon.fits" #subimage

img = Image(path)
#plt.figure(figsize=(8,8))
display_image = img.data.astype(float)

min_clip = 4000

display_image[display_image<min_clip] = min_clip + 1 # will remove the 'static' of white dots
plt.imshow(display_image, norm=LogNorm(vmin=min_clip, vmax=5000), cmap='Greys_r')

plt.scatter([x_c, x_p], [y_c,y_p]  facecolors='none', edgecolors=['b', 'r'])
plt.show()
