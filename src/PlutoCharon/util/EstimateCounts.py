# estimate the counts of pluto and charon

from Image import Image
from res import config
from res.constants import constants

i = Image(
    f'{config.output_folder}/{config.date}/{config.date}{config.index}_PC_subimage.fits'
)
print(
    (i.get_average_pixel_value() - constants[config.date + config.index]['background'])
    * 17
    * 17
)
