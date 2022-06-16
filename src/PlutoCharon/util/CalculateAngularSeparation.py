from astropy.coordinates import SkyCoord
import astropy.units as u

from res import config
from res.constants import *

# Calculate angular separation between pluto and charon and determine the distance in pixels between them

entry = constants[config.date + config.index]
plate_scale_x = entry["plate_scale_x"]  # arcseconds per pixel
plate_scale_y = entry["plate_scale_y"]  # arcseconds per pixel

pluto = SkyCoord(entry["pluto"], unit=(u.deg, u.deg), frame="icrs")
charon = SkyCoord(entry["charon"], unit=(u.deg, u.deg), frame="icrs")

# sep = pluto.separation(charon)
# pixel_sep = pluto.separation(charon).arcsecond/plate_scale

print("Horizons System Ephermeris:", config.date + config.index, entry["date_obs"])
print(
    "x separation should be "
    + str((pluto.ra - charon.ra).arcsec)
    + " arcsec = "
    + str(-(pluto.ra - charon.ra).arcsec / plate_scale_x)
    + " pixels"
)
print(
    "y separation should be "
    + str((pluto.dec - charon.dec).arcsec)
    + " arcsec = "
    + str((pluto.dec - charon.dec).arcsec / plate_scale_y)
    + " pixels"
)
