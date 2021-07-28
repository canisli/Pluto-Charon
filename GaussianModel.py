import math

from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from astropy.io import fits
from astropy.table import Table
import numpy as np

from IStar import IStar
from Image import Image, get_image_hdu_number


class GaussianModel:
    def __init__(self, center_x, center_y, image, fwhm, average_pixel_value):
        self.center_x = center_x
        self.center_y = center_y
        self.image = image
        self.fwhm = fwhm
        self.average_pixel_value = average_pixel_value
        print(image.get_pixel(639,868))
        self.a =  np.max([image.get_pixel(math.floor(center_x),math.floor(center_y)),
                                       image.get_pixel(math.ceil(center_x),math.floor(center_y)),
                                       image.get_pixel(math.floor(center_x),math.ceil(center_y)),
                                        image.get_pixel(math.ceil(center_x),math.ceil(center_y))])

        # self.a = Parameter("A", np.max([image.get_pixel(math.floor(center_x),math.floor(center_y)),
        #                                image.get_pixel(math.ceil(center_x),math.floor(center_y)),
        #                                image.get_pixel(math.floor(center_x),math.ceil(center_y)),
        #                                 image.get_pixel(math.ceil(center_x),math.ceil(center_y))]))

        self.b = average_pixel_value
        self.sigma_x2, self.sigma_y2 = fwhm/2.355**2

        self.LMparams = Parameters()
        self.LMparams.add('A', value = self.a)
        self.LMparams.add('B', value = self.b)
        self.LMparams.add('sigma_x2', value = self.sigma_x2)
        self.LMparams.add('sigma_y2', value = self.sigma_y2)

    def get_sigma(self):
        total_residual_error = 0
        dsda, dsdb, dsdsigma_x2, dsdsigma_y2 = 0

        for x in range(int(round(center_x)) - 10, int(round(center_x))+10):
                for y in range(int(round(center_y)) - 10, int(round(center_y))+10):
                    dx = center_x - x
                    dy = center_y - y
                    psf = psf(dx,dy)
                    g = self.image.get_pixel(x,y)
                    exp_term = (psf-self.b)/self.a

                    total_residual_error += psf**2
                    dsda += -2*(g-psf) * exp_term
                    dsdb += -2*(g-psf) * 1
                    dsdsigma_x2 += -2*(g-psf) * A*x**2(exp_term)/self.sigma_x2**3
                    dsdsigma_y2 += -2*(g-psf) * A*y**2(exp_term)/self.sigma_y2**3


        LMFitmin = Minimizer(psf, self.LMparams, fcn_args=(A,))
        LMFitResult = LMFitmin.minimize(method='least_square')

    def psf(params, x, y):
        return params['A'] * math.exp(-(x^2/(2*params['sigma_x2']))+(y^2/(2*params['sigma_y2']))) + params['B']


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
