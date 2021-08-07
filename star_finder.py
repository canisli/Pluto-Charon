import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.table import vstack
from astropy.visualization import simple_norm
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils.background import Background2D
from photutils.aperture import aperture_photometry
import matplotlib.pyplot as plt
from matplotlib.colors import *
import scipy.optimize as opt
from glob import glob

import sys
import math

def find_stars(im, uncertainty_im=None, minimum_pixel_size_of_stars=5, minimum_signal_to_noise=1000):
    uncertainty_im = im * 0 + 1 # a bunch of 1's
    signal_to_noise_image = im
    daofind = DAOStarFinder(
        fwhm=minimum_pixel_size_of_stars,
        threshold=minimum_signal_to_noise,
        exclude_border=True,
    )
    all_sources = daofind(
        signal_to_noise_image
    )  # it is important that we feed the image/uncertainty_image here so that our signal-to-noise cutoff works.

    print(all_sources)
    plt.figure(figsize=(8,8))
    display_image = np.copy(im)
    min_clip = 30
    display_image[display_image<min_clip] = min_clip + 1 # will remove the 'static' of white dots
    #plt.imshow(display_image, norm=LogNorm(vmin=min_clip, vmax=5000), cmap='Greys_r')
    plt.imshow(display_image, norm=LogNorm(vmin=5000, vmax=6000), cmap='Greys_r')
    plt.scatter(all_sources['xcentroid'], all_sources['ycentroid'], marker='o',facecolors='none', edgecolors='r')
    plt.show()

hdulist = fits.open("./data/4-25-2021/pluto_V.fits")
print(hdulist.info())
im = hdulist[1].data.astype(float)
find_stars(im)
