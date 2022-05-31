import astropy.units as u
from astropy.coordinates import SkyCoord

# Calculate angular separation between pluto and charon and determine the distance in pixels between them

# input data here
pluto_ra = '20h02m41.92s'
pluto_dec = '-22d30m53.3s'  # FIX TO DEGREE FORMAT ###################
charon_ra = '20h02m41.98s' 
charon_dec = '-22d30m53.2s'
plate_scale = 0.332 # arcseconds per pixel

pluto = SkyCoord(ra=pluto_ra, dec=pluto_dec, frame='icrs')
charon = SkyCoord(ra=charon_ra, dec=charon_dec, frame='icrs')

sep = pluto.separation(charon).arcsecond/plate_scale

print("The angular separation in arcseconds is ")
print(sep)
