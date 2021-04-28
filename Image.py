#!/usr/bin/python3
import math

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from IStar import IStar


def parse_file_name(file_name):
    if file_name[-5:] != ".fits":
        file_name += ".fits"
    return file_name


def image_hdu_number(hdul):
    """
    Get index of ImageHDU from HDUList
    """

    for n in [0, 1]:
        if 'EXPOSURE' in hdul[n].header:
            return n
    print("image_hdu_number: Cannot find valid HDU keywords")
    raise ValueError


def table_hdu_number(hdul):
    """
    Get index of TableHDU from HDUList
    Guaranteed that it is the last HDU
    """

    return len(hdul) - 1


def get_star_attribute(data, index):
    if data[index] == 0.0:
        return "N/A"
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
                self.data = hdul[image_hdu_number(hdul)].data

            self.height = len(self.data)
            self.width = len(self.data[0])
        else:  # create blank Image
            self.based_on_existing_file = False
            self.file_name = None
            self.width = width
            self.height = height

            self.data = np.zeros((height, width))  # create 2D array full of zeros
        print("Created " + str(self))

    def is_inside(self, x, y):
        if self.height is None or self.width is None:
            return False  # ?
        if y < 0 or y > self.height or x < 0 or x > self.width:
            return False
        return True

    def get_pixel(self, x, y):
        return self.data[y - 1][x - 1]

    def set_pixel(self, x, y, val):
        if not self.is_inside(x, y):
            print("set_pixel: coordinate is out of range")
            raise IndexError
        self.data[y - 1][x - 1] = val

    def set_pixel_range(self, x1, y1, x2, y2, val):
        if not self.is_inside(x1, y1) or not self.is_inside(x2, y2):
            print("set_pixel_range: coordinate is out of range")
            raise IndexError
        for x in range(x1, x2 + 1):  # draw a rectangle
            for y in range(y1, y2 + 1):
                self.set_pixel(x, y, val)

    def write_fits(self, file_name_string=None):
        """
        Overwrite existing FITS file or Create new FITS file from Image
        If overwriting the FITS file that current Image is based on, then file_name_string is not required
        If Image is not based on existing file, file_name_string is required
        """

        if file_name_string is None:
            if not self.based_on_existing_file:
                print("write_fits: Image is not based on existing file. A specified file name is required")
                raise ValueError
            else:  # overwrite existing file
                with fits.open(self.file_name) as hdul:
                    hdul[image_hdu_number(hdul)].data = self.data
                    hdul.writeto(self.file_name, overwrite=True)

        else:  # create new file
            file_name_string = parse_file_name(file_name_string)
            if self.based_on_existing_file:
                with fits.open(self.file_name) as hdul:
                    hdul[image_hdu_number(hdul)].data = self.data
                    hdul[image_hdu_number(hdul)].header['EXPOSURE'] = 0
                    hdul.writeto(file_name_string, overwrite=True)
            else:
                primary_hdu = fits.PrimaryHDU()
                image_hdu = fits.ImageHDU(self.data.tolist())
                image_hdu.header['EXPOSURE'] = 0
                new_hdul = fits.HDUList([primary_hdu, image_hdu])
                self.file_name = parse_file_name(file_name_string)
                new_hdul.writeto(self.file_name, overwrite=True)

    def get_stars(self):
        """
        Returns a list of IStar objects from the HDUL's TableHDU information
        """
        if self.file_name is None:
            return []
        with fits.open(self.file_name) as hdul:
            table = Table(hdul[table_hdu_number(hdul)].data)
            star_data = table.as_array()
            stars = []
            for i in range(len(table)):
                stars.append(
                    IStar(star_name=get_star_attribute(star_data[i], 0),
                          x=get_star_attribute(star_data[i], 1),
                          y=get_star_attribute(star_data[i], 2),
                          magnitude=get_star_attribute(star_data[i], 5),
                          counts=get_star_attribute(star_data[i], 7)))
        return stars

    def plot_intensity_profile(self, center_x, center_y):
        distance_from_star = []  # x
        values = []  # y
        for x in range(round(center_x) - 20, round(center_x) + 20 + 1):
            for y in range(round(center_y) - 20, round(center_y) + 20 + 1):
                if self.is_inside(x, y):
                    distance_from_star.append(distance(x, y, center_x, center_y))
                    values.append(self.get_pixel(x, y))
        plt.plot(distance_from_star, values)
        plt.xlabel("Distance to star")
        plt.ylabel("Pixel Value")
        plt.savefig(self.file_name + ".png")
        plt.show()

    def __str__(self):
        if self.file_name is not None:
            return ("Image " + str(self.file_name) + ": "
                    + str(self.width) + " x " + str(self.height))
        else:
            return ("Image: "
                    + str(self.width) + " x " + str(self.height))

    def __del__(self):
        print("Destroyed " + str(self))


def log_stars(stars, file_name):
    f = open(file_name, "w")
    for star in stars:
        f.write(str(star) + "\n")
    f.close()


def main():
    # image = Image(file_name="hello")
    # image2 = Image(width=800, height=500)
    #
    # image.set_pixel_range(0, 0, 800, 800, 33300)
    # print(image.get_pixel(50, 50))
    #
    # image.write_fits("hello")
    # #  image.write_fits()
    image = Image(file_name="u-aur_V.fits")
    log_stars(image.get_stars(), "u-aur_V_stars.txt")
    image.plot_intensity_profile(1006.5, 281.26)


if __name__ == "__main__":
    main()
