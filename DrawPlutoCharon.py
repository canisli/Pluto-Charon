# Utility to draw circles around Pluto and Charon given their locations

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from Image import Image
import config

"""
Bad solution to 4-25-2021
x_p = 9.52835602
y_p = 9.44556182
x_c = 9.52135037
y_c = 9.43923984

"""
"""
Good solution to 5-13-2021
x_p = 8.93626205723376
y_p = 9.20472722052157
x_c = 9.434444167685312
y_c = 8.505905815003175
"""
"""
Another good solution to 5-13-2021
x_p = 8.933036982803547
y_p = 9.209683430338993
x_c = 9.32162087423003
y_c =8.66271714521153
"""

x_p = 8.933036982803547
y_p = 9.209683430338993
x_c = 9.32162087423003
y_c = 8.66271714521153


path = config.data_folder  + config.date + "/plutocharon.fits" #subimage
print(path)
img = Image(path)
plt.figure(figsize=(8,8))
display_image = img.data.astype(float)

# min_clip = 4000 4-25-2021
min_clip = 1300
vmax = 3000

display_image[display_image<min_clip] = min_clip + 1 # will remove the 'static' of white dots
plt.imshow(display_image, norm=LogNorm(vmin=min_clip, vmax=vmax), cmap='Greys_r')

plt.scatter([x_c, x_p], [y_c,y_p] , facecolors='none', edgecolors=['b', 'r'])
plt.show()
