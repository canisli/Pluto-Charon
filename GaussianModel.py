import math

from lmfit import Parameter
from astropy.io import fits
from astropy.table import Table
import numpy as np

from IStar import IStar
from Image import Image
from Image import get_image_hdu_number


class GaussianModel:
    def __init__(self, center_x, center_y, image, fwhm, average_pixel_value):
        self.center_x = center_x
        self.center_y = center_y
        self.image = image
        self.fwhm = fwhm
        self.average_pixel_value = average_pixel_value
        print(image.get_pixel(639,868))
        self.a = Parameter("A", np.max([image.get_pixel(math.floor(center_x),math.floor(center_y)),
                                       image.get_pixel(math.ceil(center_x),math.floor(center_y)),
                                       image.get_pixel(math.floor(center_x),math.ceil(center_y)),
                                        image.get_pixel(math.ceil(center_x),math.ceil(center_y))]))

        print(self.a.value)
        print(([image.get_pixel(math.floor(center_x),math.floor(center_y)),
                                       image.get_pixel(math.ceil(center_x),math.floor(center_y)),
                                       image.get_pixel(math.floor(center_x),math.ceil(center_y)),
                                        image.get_pixel(math.ceil(center_x),math.ceil(center_y))]))
        self.b = Parameter("B", average_pixel_value)
        self.sigma_x2 = Parameter(
            "sigma_x2", fwhm/2.355**2
        )
        self.sigma_y2 = Parameter(
            "sigma_y2", fwhm/2.355**2
        )

    def get_sigma(self):
        pass


def main():
    path = "./data/" + "4-25-2021" + "/pluto_V.fits"
    starlist_path = "./out/" + "4-25-2021.csv"
    starlist = Table.read(starlist_path, format='csv')

    image = Image(path)
    hdul = fits.open(path)

    fwhm_arc = 3.5  # full width half maximum in arcseconds
    fwhm = fwhm_arc / hdul[get_image_hdu_number(hdul)].header["CDELT1"] # fwhm in pixels


    # arbitrary star
    print(starlist[0])
    star = IStar(table_row=starlist[0])
    gm = GaussianModel(star.x, star.y, image, fwhm, image.get_average_pixel_value())


if __name__ == "__main__":
    main()
