from astropy.coordinates import SkyCoord
import astropy.units as u

from res import config
from res.constants import *


# Calculate angular separation between pluto and charon and determine the distance in pixels between them
for index in config.indices:
    entry = constants[config.date + index]
    plate_scale_x = entry['plate_scale_x']  # arcseconds per pixel
    plate_scale_y = entry['plate_scale_y']  # arcseconds per pixel

    pluto = SkyCoord(entry['pluto'], unit=(u.deg, u.deg), frame='icrs')
    charon = SkyCoord(entry['charon'], unit=(u.deg, u.deg), frame='icrs')

    # sep = pluto.separation(charon)
    # pixel_sep = pluto.separation(charon).arcsecond/plate_scale

    print(
        f'From Horizons System Ephermeris: {config.date}{config.index} at {entry["date_obs"]}'
    )
    print(
        f'x separation should be {(pluto.ra - charon.ra).arcsec} arcsec =  {-(pluto.ra - charon.ra).arcsec / plate_scale_x} pixels'
    )
    print(
        f'x separation should be {(pluto.dec - charon.dec).arcsec} arcsec =  {-(pluto.dec- charon.dec).arcsec / plate_scale_y} pixels'
    )
