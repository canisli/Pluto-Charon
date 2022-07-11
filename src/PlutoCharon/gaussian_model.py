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
    
    def __init__(self, psf_setup_data):
        self.psf_setup_data = psf_setup_data

    def get_params(self):
        self.LMFitResult = minimize(
            self.psf_error,
            self.LMparams,
            args=(),
            method='least_squares',
            max_nfev=90000, #14000,
        )
        return self.LMFitResult.params

    def psf(self):
        pass

    def psf_error(self, LMparams):
        errors = [self.image.get_pixel(x, y) - self.psf(LMparams, x, y) 
                  for x in range(self.image.width)
                  for y in range(self.image.height)]
        return errors

    def get_result(self):
        return self.LMFitResult.message


# Single Gaussian Model for stars in the image
# Estimates A, B, sigma_x2, sigma_y2, and the center of the star
class StarGaussian(GaussianModel):
    def __init__(self, psf_setup_data):
        super().__init__(psf_setup_data)
        center_x = self.psf_setup_data['star_x']
        center_y = self.psf_setup_data['star_y']
        self.image = self.psf_setup_data['image']
        a = np.max(
            [  # four pixels around center
                self.image.get_pixel(math.floor(center_x), math.floor(center_y)),
                self.image.get_pixel(math.ceil(center_x), math.floor(center_y)),
                self.image.get_pixel(math.floor(center_x), math.ceil(center_y)),
                self.image.get_pixel(math.ceil(center_x), math.ceil(center_y)),
            ]
        )
        sigma = (psf_setup_data['fwhm'] / 2.355)

        self.LMparams = Parameters()
        self.LMparams.add('xc', value=psf_setup_data['subimage'].width / 2 - 0.5)
        self.LMparams.add('yc', value=psf_setup_data['subimage'].height / 2 - 0.5)
        self.LMparams.add('a', value=a)
        self.LMparams.add('bg', value=psf_setup_data['avg_pixel_val'])
        self.LMparams.add('sigma_x', value=sigma)
        self.LMparams.add('sigma_y', value=sigma)

    def get_params(self):
        return super().get_params()

    def psf(self, LMparams, x, y):
        a = LMparams['a']
        b = LMparams['bg']
        xc = LMparams['xc']
        yc = LMparams['yc']
        sigma_x = LMparams['sigma_x']
        sigma_y = LMparams['sigma_y']
        return psf(x-xc, y-yc, sigma_x, sigma_y, 0, a, b)


class DoubleGaussian(GaussianModel):
    """
    Double Gaussian Model for Pluto Charon blob
    Solves for coordinates of Pluto and Charon, amplitude of their PSFs, 
        background, and ellipse orientation angle
    """
    def __init__(self, psf_setup_data):
        super().__init__(psf_setup_data)
        self.LMparams = Parameters()
        self.image = self.psf_setup_data['image']
        self.LMparams.add('x_p', value=self.psf_setup_data['x_p'], min=0, max=self.image.width)
        self.LMparams.add('y_p', value=self.psf_setup_data['y_p'], min=0, max=self.image.height)
        self.LMparams.add('dx', value=self.psf_setup_data['dx'], min=-10, max=10)
        self.LMparams.add('dy', value=self.psf_setup_data['dy'], min=-10, max=10)
        self.LMparams.add('a_p', value=self.psf_setup_data['init_Ap'], min=10, max=1000)
        self.LMparams.add('a_c', value=self.psf_setup_data['init_Ac'], min=10, max=1000)
        self.LMparams.add('bg', value=self.psf_setup_data['init_background'], min=400, max=600)
        self.LMparams.add('theta', value=self.psf_setup_data['theta'])
        self.LMparams.add(
            'sigma_x', value=self.psf_setup_data['sigma_x'], vary=True, min=1, max=100
        )
        self.LMparams.add(
            'sigma_y', value=self.psf_setup_data['sigma_y'], vary=True, min=1, max=100
        )

    def get_params(self):
        """
        Returns all seven params {x_c, y_c, x_p,y_p, A_p, A_c, bg, theta}
        """
        return super().get_params()

    def psf(self, LMparams, x, y):
        a_c = LMparams['a_c']
        a_p = LMparams['a_p']
        bg = LMparams['bg']
        sigma_x = LMparams['sigma_x']
        sigma_y = LMparams['sigma_y']
        theta = LMparams['theta']
        x_p = LMparams['x_p']
        y_p = LMparams['y_p']
        dx = LMparams['dx']
        dy = LMparams['dy']

        return (
            bg + psf(x-x_p, y-y_p, sigma_x, sigma_y, theta, a_p, 0)
              + psf(x-x_p+dx, y-y_p+dy, sigma_x, sigma_y, theta, a_c, 0)
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
