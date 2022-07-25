import math
import logging
import sys

from lmfit import minimize, Parameters
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger('gaussian')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def psf(x, y, sigma_x, sigma_y, angle, peak, background):
    cosa = math.cos(angle)
    sina = math.sin(angle)
    return background + peak * math.exp(
        -(
            (x * cosa + y * sina) ** 2 / (2 * sigma_x**2)
            + (x * sina - y * cosa) ** 2 / (2 * sigma_y**2)
        )
    )


class GaussianModel:
    """
    Superclass for running least squares fitting on PSF function
    """

    def __init__(self, psfinit, image):
        self.psfinit = psfinit
        self.image = image
        self.LMparams = Parameters()
        for param in psfinit:
            self.LMparams.add(param, psfinit[param], vary=False)
    
    def set_vary(self, params_to_vary):
        for p in params_to_vary:
            self.LMparams[p].vary = True

    def set_limits(self, param, min, max):
        self.LMparams[param].min = min
        self.LMparams[param].max = max

    def run_minimizer(self, method='least_squares'):
        self.LMFitResult = minimize(
            self.psf_error,
            self.LMparams,
            args=(),
            method=method,
            max_nfev=14000,
        )
        return self.LMFitResult.params

    def psf(self):
        pass

    def psf_error(self, LMparams):
        errors = [
            self.image.get_pixel(x, y) - self.psf(LMparams, x, y)
            for x in range(self.image.width)
            for y in range(self.image.height)
        ]
        return errors

    def get_result(self):
        return self.LMFitResult.message


# Single Gaussian Model for stars in the image
# Estimates A, B, sigma_x2, sigma_y2, and the center of the star
class SingleGaussian(GaussianModel):
    def psf(self, LMparams, x, y):
        a = LMparams['a']
        bg = LMparams['bg']
        xc = LMparams['xc']
        yc = LMparams['yc']
        sigma_x = LMparams['sigma_x']
        sigma_y = LMparams['sigma_y']
        theta = LMparams['theta'].value
        return psf(x - xc, y - yc, sigma_x, sigma_y, theta, a, bg)


class DoubleGaussian(GaussianModel):
    """
    Double Gaussian Model for Pluto Charon blob
    Solves for coordinates of Pluto and Charon, amplitude of their PSFs,
        background, and ellipse orientation angle
    """

    def psf(self, LMparams, x, y):
        a_p = LMparams['a_p']
        a_c = a_p * LMparams['rel_flux']
        bg = LMparams['bg']
        sigma_x = LMparams['sigma_x']
        sigma_y = LMparams['sigma_y']
        theta = LMparams['theta']
        x_p = LMparams['x_p']
        y_p = LMparams['y_p']
        dx = LMparams['dx']
        dy = LMparams['dy']

        return (
            bg
            + psf(x - x_p, y - y_p, sigma_x, sigma_y, theta, a_p, 0)
            + psf(x - x_p + dx, y - y_p + dy, sigma_x, sigma_y, theta, a_c, 0)
        )


def get_params_from_file(file_path):
    fittings = Table.read(file_path, format='ascii.fixed_width_two_line')
    return fittings


def plot_params(fittings, center_x, center_y):
    # deprecated
    center_dist = []
    for x, y in zip(fittings['x'], fittings['y']):  # iterate in parallel
        center_dist.append(math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
    log.info(center_dist, fittings['sigma_x2'])
    plt.title('Sigma_x2 as function from distance from center')
    plt.scatter(np.array(center_dist), np.array(fittings['sigma_x2']), linestyle='None')
    plt.show()
    plt.figure()
    plt.title('Sigma_y2 as function from distance from center')
    plt.scatter(np.array(center_dist), np.array(fittings['sigma_y2']), linestyle='None')
    plt.show()
    log.info('sigma_x2 ' + str(np.average(fittings['sigma_x2'])))
    log.info('sigma_y2 ' + str(np.average(fittings['sigma_y2'])))


def compute_avg(file_path, upper_bound=10):
    """compute the average sigma_x2 and sigma_y2. ignores vals over threshold"""
    fittings = get_params_from_file(file_path)

    avg_sigma_x2 = np.average([x for x in fittings['sigma_x2'] if x < upper_bound])
    avg_sigma_y2 = np.average([x for x in fittings['sigma_y2'] if x < upper_bound])
    return (avg_sigma_x2, avg_sigma_y2)
