# Utility to draw circles around Pluto and Charon given their locations

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from Image import Image
from res import config

x_p = 4.696478200520507 
y_p = 8.889831536002246
x_c = 10.212682325016397
y_c = 9.984729078747533


print("The pixel distance between pluto and charon is " + str(np.sqrt((x_p-x_c)**2+(y_p-y_c)**2)))


path = config.output_folder + config.date + "/" + config.date+ "_PC_subimage.fits"  # subimage
print(path)
img = Image(path)
plt.figure(figsize=(8, 8))
display_image = img.data.astype(float)

# min_clip = 4000 4-25-2021
min_clip = 505
vmax = 517

display_image[display_image < min_clip] = (
    min_clip + 1
)  # will remove the 'static' of white dots
plt.imshow(display_image, norm=LogNorm(vmin=min_clip, vmax=vmax), cmap="Greys_r")

plt.scatter([x_c, x_p], [y_c, y_p], facecolors="none", edgecolors=["b", "r"])
ax = plt.gca()
ax.invert_yaxis()
plt.show()
