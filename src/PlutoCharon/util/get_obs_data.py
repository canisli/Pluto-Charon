from astropy.io import fits

import image
from res import config

hdul = fits.open(config.data_folder + config.date + '/pluto' + config.index + '.fits')

print('Date-OBS, plate scale x and plate scale y:')
print(hdul[Image.get_image_hdu_number(hdul)].header['DATE-OBS'])
print(hdul[Image.get_image_hdu_number(hdul)].header['CDELT1'])
print(hdul[Image.get_image_hdu_number(hdul)].header['CDELT2'])
