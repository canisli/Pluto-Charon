import math
import logging
import sys

from lmfit import minimize, Parameters
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger('GaussianModel')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def psf(x, y, sigma_x2, sigma_y2, angle, peak, background):
    cosa = math.cos(angle)
    sina = math.sin(angle)
    return background + peak * math.exp(
        -(
            (x * cosa + y * sina) ** 2 / (2 * sigma_x2)
            + (x * sina - y * cosa) ** 2 / (2 * sigma_y2)
        )
    )

class GaussianModel:
    """
    Superclass for running least squares fitting on PSF function
    """
    
    def __init__(self, PSFSetupData):
        self.PSFSetupData = PSFSetupData
        self.image = self.PSFSetupData['subimage']

    def get_params(self):
        LMFitResult = minimize(
            self.psf_error,
            self.LMparams,
            args=(),
            method='least_squares',
            max_nfev=14000,
        )
        return LMFitResult.params

    def psf(self):
        pass

    def psf_error(self):
        pass


# Single Gaussian Model for stars in the image
# Estimates A, B, sigma_x2, sigma_y2, and the center of the star
class StarGaussian(GaussianModel):
    def __init__(self, PSFSetupData):
        super().__init__(PSFSetupData)
        center_x = PSFSetupData['star_x']
        center_y = PSFSetupData['star_y']
        image = PSFSetupData['orig_image']
        a = np.max(
            [  # four pixels around center
                image.get_pixel(math.floor(center_x), math.floor(center_y)),
                image.get_pixel(math.ceil(center_x), math.floor(center_y)),
                image.get_pixel(math.floor(center_x), math.ceil(center_y)),
                image.get_pixel(math.ceil(center_x), math.ceil(center_y)),
            ]
        )
        sigma2 = (PSFSetupData['fwhm'] / 2.355) ** 2

        self.LMparams = Parameters()
        self.LMparams.add('xc', value=PSFSetupData['subimage'].width / 2 - 0.5)
        self.LMparams.add('yc', value=PSFSetupData['subimage'].height / 2 - 0.5)
        self.LMparams.add('a', value=a)
        self.LMparams.add('bg', value=PSFSetupData['avg_pixel_val'])
        self.LMparams.add('sigma_x2', value=sigma2)
        self.LMparams.add('sigma_y2', value=sigma2)

    def get_params(self):
        return super().get_params()

    def psf(self, LMparams, x, y):
        a = LMparams['a']
        b = LMparams['bg']
        xc = LMparams['xc']
        yc = LMparams['yc']
        sigma_x2 = LMparams['sigma_x2']
        sigma_y2 = LMparams['sigma_y2']
        return psf(x-xc, y-yc, sigma_x2, sigma_y2, 0, a, b)

    def psf_error(self, LMparams):
        subimage = self.image
        xs = [x for x in range(subimage.width) for y in range(subimage.height)]
        ys = [y for x in range(subimage.width) for y in range(subimage.height)]
        vals = [
            subimage.get_pixel(x, y)
            for x in range(subimage.width)
            for y in range(subimage.height)
        ]
        errors = [vals[i] - self.psf(LMparams, xs[i], ys[i]) for i in range(len(vals))]
        return errors


class PlutoCharonGaussian(GaussianModel):
    """
    Double Gaussian Model for Pluto Charon blob
    Solves for coordinates of Pluto and Charon, amplitude of their PSFs, 
        background, and ellipse orientation angle
    
    """
    def __init__(self, PSFSetupData):
        super().__init__(PSFSetupData)
        self.LMparams = Parameters()
        self.LMparams.add('x_0p', value=self.PSFSetupData['x_0p'], min=0, max=19)
        self.LMparams.add('y_0p', value=self.PSFSetupData['y_0p'], min=0, max=19)
        self.LMparams.add('x_0c', value=self.PSFSetupData['x_0c'], min=0, max=19)
        self.LMparams.add('y_0c', value=self.PSFSetupData['y_0c'], min=0, max=19)
        self.LMparams.add('a_p', value=self.PSFSetupData['init_Ap'])
        self.LMparams.add('a_c', value=self.PSFSetupData['init_Ac'])
        self.LMparams.add('bg', value=self.PSFSetupData['init_background'], min=0)
        self.LMparams.add('theta', value=self.PSFSetupData['theta'], min = 0, max=2*np.pi)
        self.LMparams.add(
            'sigma_x2', value=self.PSFSetupData['sigma_x2'], min=1, max=100
        )
        self.LMparams.add(
            'sigma_y2', value=self.PSFSetupData['sigma_y2'], min=1, max=100
        )

    def get_params(self):
        """
        Returns all seven params {x_0c, y_0c, x_0p,y_0p, A_p, A_c, bg, theta}
        """
        return super().get_params()

    def psf(self, LMparams, x, y):
        a_c = LMparams['a_c']
        a_p = LMparams['a_p']
        bg = LMparams['bg']
        sigma_x2 = LMparams['sigma_x2']
        sigma_y2 = LMparams['sigma_y2']
        theta = LMparams['theta']
        x_c = LMparams['x_0c']
        y_c = LMparams['y_0c']
        x_p = LMparams['x_0p']
        y_p = LMparams['y_0p']

        return (
            bg + psf(x-x_p, y-y_p, sigma_x2, sigma_y2, theta, a_p, 0)
              + psf(x-x_c, y-y_c, sigma_x2, sigma_y2, theta, a_c, 0)
        )

    def psf_error(self, LMparams):
        errors = [self.image.get_pixel(x, y) - self.psf(LMparams, x, y) 
                  for x in range(self.image.width)
                  for y in range(self.image.height)]
        return errors


def locate_pluto_charon(PlutoCharonSetupData):
    """
    Run PlutoCharonGaussian over multiple initial conditions and
    return dict of results
    """
    locations = Table({
        'x_p': [], 'y_p': [],
        'x_c': [], 'y_c': [],
        'dx_pixel': [], 'dy_pixel': [],
        'sigma_x2': [], 'sigma_y2': [],
        'theta': [],
    })

    scale = 2
    dx_p = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]) * scale
    dy_p = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]) * scale
    dx_c = np.array([-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1]) * scale
    dy_c = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]) * scale
    1,
    # guess_x = PlutoCharonSetupData['subimage'].width/2
    # guess_y = PlutoCharonSetupData['subimage'].height/2
    guess_x = 9
    guess_y = 9
    for i in range(len(dx_p)):  # test the four locations as per the convergence diagram
        PlutoCharonSetupData['x_0p'] = guess_x + dx_p[i]
        PlutoCharonSetupData['y_0p'] = guess_y + dy_p[i]
        PlutoCharonSetupData['x_0c'] = guess_x + dx_c[i]
        PlutoCharonSetupData['y_0c'] = guess_y + dy_c[i]
        pluto_charon = PlutoCharonGaussian(PlutoCharonSetupData)
        params = pluto_charon.get_params()
        log.info('Locations after fitting')
        log.info(f'Sigma2 {params["sigma_x2"].value}, {params["sigma_y2"].value}')
        log.info(f'Pluto {params["x_0p"].value}, {params["y_0p"].value}')
        log.info(f'Charon {params["x_0c"].value}, {params["y_0c"].value}')
        
        locations.add_row([params['x_0p'].value, params['y_0p'].value,
                          params['x_0c'].value, params['y_0c'].value,
                          params['x_0p'].value - params['x_0c'].value,
                          params['y_0p'].value - params['y_0c'].value,
                          params['sigma_x2'].value, params['sigma_y2'].value,
                          params['theta'].value])

    return locations


def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_params_from_file(file_path):
    fittings = Table.read(file_path, format='ascii.fixed_width_two_line')
    return fittings


def plot_params(fittings, center_x, center_y):
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
