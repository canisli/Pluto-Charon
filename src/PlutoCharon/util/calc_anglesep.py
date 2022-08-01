from astropy.coordinates import SkyCoord
import astropy.units as u

plate_scale = .34

# 6-9-2022
pluto = SkyCoord('300.43646 -22.59674', unit=(u.deg, u.deg), frame='icrs')
charon = SkyCoord('300.43624 -22.59677', unit=(u.deg, u.deg), frame='icrs')

# 7-7-2022
pluto = SkyCoord('299.82808 -22.76706', unit=(u.deg, u.deg), frame='icrs')
charon = SkyCoord('299.82820 -22.76691', unit=(u.deg, u.deg), frame='icrs')

print(
    f'Pluto is {(pluto.ra - charon.ra).arcsec} arcsec =  {(pluto.ra - charon.ra).arcsec / plate_scale} pixels to the right of Charon'
)
print(
    f'Pluto is {(pluto.dec - charon.dec).arcsec} arcsec =  {(pluto.dec- charon.dec).arcsec / plate_scale} pixels above Charon'
)
