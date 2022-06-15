from astropy.coordinates import SkyCoord
import astropy.units as u

from res import config
from res.constants import *

# Calculate angular separation between pluto and charon and determine the distance in pixels between them

entry = constants[config.date + config.index]
#plate_scale = entry["plate_scale"]  # arcseconds per pixel

pluto = SkyCoord(entry["pluto"], unit=(u.hourangle, u.deg), frame="icrs")
charon = SkyCoord(entry["charon"], unit=(u.hourangle, u.deg), frame="icrs")

sep = pluto.separation(charon)
#pixel_sep = pluto.separation(charon).arcsecond/plate_scale

print("Horizons System Ephermeris")
print("The angular separation in pixels should be ")
#print(pixel_sep)
print("The angular separation in arcseconds should be ")
print(sep)

