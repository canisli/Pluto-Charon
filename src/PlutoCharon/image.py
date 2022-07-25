import math
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from istar import IStar

log = logging.getLogger('image')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_file_name(file_name):
    if file_name[-3:] == '.fz':
        return file_name
    if file_name[-5:] != '.fits':
        file_name += '.fits'
    return file_name


def get_image_hdu_number(hdul):
    """
    Get index of ImageHDU from HDUList
    """
    for n in [0, 1]:
        if 'EXPOSURE' in hdul[n].header:
            return n
    log.info('get_image_hdu_number: Cannot find valid HDU keywords')
    raise ValueError


def get_table_hdu_number(hdul):
    """
    Get index of TableHDU from HDUList
    Guaranteed that it is the last HDU
    """

    return len(hdul) - 1


def get_star_attribute(data, index):
    # Check if each value is valid
    flag = data[9]
    if index == 0:  # star name
        if not (flag & 0x20):
            return 'N/A'
    if index == 1 or index == 2:  # X, Y
        if not (flag & 0x01):
            return 'N/A'
    if index == 3 or index == 4:  # DEC RA
        if not (flag & 0x010):
            return 'N/A'
    if index == 5:  # MAG
        if not (flag & 0x02):
            return 'N/A'
    if index == 6:  # BKGD
        if not (flag & 0x04):
            return 'N/A'
    if index == 7:  # COUNTS
        if not (flag & 0x08):
            return 'N/A'
    if index == 8:  # PHOTOMETRY
        if not (flag & 0x40):
            return 'N/A'
    return data[index]


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class Image:
    def __init__(self, file_name=None, width=None, height=None):
        """
        Construct Image object
        Either input existing FITS file or width and height of new blank Image
        """

        if file_name is not None:  # open existing file
            self.based_on_existing_file = True
            file_name = parse_file_name(file_name)
            self.file_name = file_name
            with fits.open(self.file_name) as hdul:
                self.data = hdul[get_image_hdu_number(hdul)].data

            self.height = len(self.data)
            self.width = len(self.data[0])
        else:  # create blank Image
            self.based_on_existing_file = False
            self.file_name = None
            self.width = width
            self.height = height

            # create 2D array full of zeros
            self.data = np.zeros((height, width))
        log.debug('Created ' + str(self))

    def is_inside(self, x, y):
        if self.height is None or self.width is None:
            return False  # ?
        if y < 0 or y > self.height or x < 0 or x > self.width:
            return False
        return True

    def max_pixel(self):
        return np.max(self.data)

    def get_pixel(self, x, y):
        return self.data[y][x]  # 0 indexed

    def subimage(self, center_x, center_y, subimage_width, subimage_height):
        center_x = int(center_x)
        center_y = int(center_y)
        subimage = Image(width=subimage_width, height=subimage_height)
        for x in range(-int(subimage_width / 2), int((subimage_width + 1) / 2)):
            for y in range(-int(subimage_height / 2), int((subimage_height + 1) / 2)):
                subimage.set_pixel(
                    (int)(subimage_width / 2) + x,
                    (int)(subimage_height / 2) + y,
                    self.get_pixel(center_x + x, center_y + y),
                )
        return subimage

    def set_pixel(self, x, y, val):
        if not self.is_inside(x, y):
            log.info('set_pixel: coordinate is out of range')
            raise IndexError
        self.data[y][x] = val

    def set_pixel_range(self, x1, y1, x2, y2, val):
        if not self.is_inside(x1, y1) or not self.is_inside(x2, y2):
            log.info('set_pixel_range: coordinate is out of range')
            raise IndexError
        for x in range(x1, x2 + 1):  # draw a rectangle
            for y in range(y1, y2 + 1):
                self.set_pixel(x, y, val)

    def get_average_pixel_value(self):
        return np.average(self.data)

    def write_fits(self, file_name_string=None):
        """
        Overwrite existing FITS file or Create new FITS file from Image
        If overwriting the FITS file that current Image is based on, then file_name_string is not required
        If Image is not based on existing file, file_name_string is required
        """

        if file_name_string is None:
            if not self.based_on_existing_file:
                log.info(
                    'write_fits: Image is not based on existing file. A specified file name is required'
                )
                raise ValueError
            else:  # overwrite existing file
                with fits.open(self.file_name) as hdul:
                    hdul[get_image_hdu_number(hdul)].data = self.data
                    hdul.writeto(self.file_name, overwrite=True)

        else:  # create new file
            file_name_string = parse_file_name(file_name_string)
            if self.based_on_existing_file:
                with fits.open(self.file_name) as hdul:
                    hdul[get_image_hdu_number(hdul)].data = self.data
                    hdul[get_image_hdu_number(hdul)].header['EXPOSURE'] = 0
                    hdul.writeto(file_name_string, overwrite=True)
            else:
                primary_hdu = fits.PrimaryHDU()
                image_hdu = fits.ImageHDU(self.data.tolist())
                image_hdu.header['EXPOSURE'] = 0
                new_hdul = fits.HDUList([primary_hdu, image_hdu])
                self.file_name = parse_file_name(file_name_string)
                new_hdul.writeto(self.file_name, overwrite=True)

    def get_image_hdu_value(self, name):
        """
        Get value of certain value from image hdu
        """
        with fits.open(self.file_name) as hdul:
            try:
                return hdul[get_image_hdu_number(hdul)].header[name.upper()]
            except KeyError:
                log.info('ERROR: Keyword ' ' + str(name) + ' ' not found')

    def get_stars(self):
        """
        Returns a list of IStar objects from the HDUL's TableHDU information
        """
        if self.file_name is None:
            return []
        with fits.open(self.file_name) as hdul:
            table = Table(hdul[get_table_hdu_number(hdul)].data)
            star_data = table.as_array()
            stars = []
            for i in range(len(table)):
                stars.append(
                    IStar(
                        star_name=get_star_attribute(star_data[i], 0),
                        x=get_star_attribute(star_data[i], 1),
                        y=get_star_attribute(star_data[i], 2),
                        magnitude=get_star_attribute(star_data[i], 5),
                        counts=get_star_attribute(star_data[i], 7),
                    )
                )
        return stars

    def save_starlist(self, out_path):
        stars = self.get_stars()
        output = {'name': [], 'x': [], 'y': [], 'mag': [], 'counts': []}
        for star in stars:
            output['name'].append(star.star_name)
            output['x'].append(star.x)
            output['y'].append(star.y)
            output['mag'].append(star.magnitude)
            output['counts'].append(star.counts)
        Table(output).write(
            out_path, format='ascii.fixed_width_two_line', overwrite=True
        )

    def plot_intensity_profile(self, center_x, center_y):
        distance_from_star = []  # x
        values = []  # y
        for x in range(round(center_x) - 20, round(center_x) + 20 + 1):
            for y in range(round(center_y) - 20, round(center_y) + 20 + 1):
                if self.is_inside(x, y):
                    distance_from_star.append(distance(x, y, center_x, center_y))
                    values.append(self.get_pixel(x, y))
        plt.plot(distance_from_star, values, 'o')
        plt.xlabel('Distance to star')
        plt.ylabel('Pixel Value')
        plt.savefig(self.file_name[0:-5] + '_graph.png')
        plt.show()

    def __str__(self):
        if self.file_name is not None:
            return f'Image {self.file_name}: {self.width} x {self.height}'
        else:
            return f'Image: {self.width} x {self.height}'

    def __del__(self):
        log.debug(f'Destroyed {self}')
