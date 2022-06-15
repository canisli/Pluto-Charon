import math

from lmfit import minimize, Parameters, fit_report
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from res import config
from res.constants import *

# Superclass
# Contains method to estimate parameters with least squares fitting.
# PSF and PSF_error functions need to be implemented.
class GaussianModel:
    def __init__(self, PSFSetupData):
        self.PSFSetupData = PSFSetupData
        self.subimage = self.PSFSetupData["subimage"]

    def get_params(self):
        LMFitResult = minimize(
            self.psf_error,
            self.LMparams,
            args=(),
            method="least_squares",
            max_nfev=14000,
        )
        if config.do_debugging_for_gaussian:
            print(fit_report(LMFitResult))
        self.LMparams = LMFitResult.params
        return self.LMparams

    def psf(self):
        pass

    def psf_error(self):
        pass


# Single Gaussian Model for stars in the image
# Estimates A, B, sigma_x2, sigma_y2, and the center of the star
class StarGaussian(GaussianModel):
    def __init__(self, PSFSetupData):
        super().__init__(PSFSetupData)
        center_x = PSFSetupData["star_x"]
        center_y = PSFSetupData["star_y"]
        image = PSFSetupData["orig_image"]
        a = np.max(
            [  # four pixels around center
                image.get_pixel(math.floor(center_x), math.floor(center_y)),
                image.get_pixel(math.ceil(center_x), math.floor(center_y)),
                image.get_pixel(math.floor(center_x), math.ceil(center_y)),
                image.get_pixel(math.ceil(center_x), math.ceil(center_y)),
            ]
        )
        sigma2 = (PSFSetupData["fwhm"] / 2.355) ** 2

        self.LMparams = Parameters()
        self.LMparams.add("xc", value=PSFSetupData["subimage"].width / 2 - 0.5)
        self.LMparams.add("yc", value=PSFSetupData["subimage"].height / 2 - 0.5)
        self.LMparams.add("a", value=a)
        self.LMparams.add("b", value=PSFSetupData["avg_pixel_val"])
        self.LMparams.add("sigma_x2", value=sigma2)
        self.LMparams.add("sigma_y2", value=sigma2)

    def get_params(self):
        return super().get_params()

    def psf(self, LMparams, x, y):
        a = LMparams["a"].value
        b = LMparams["b"].value
        xc = LMparams["xc"].value
        yc = LMparams["yc"].value
        dx = x - xc
        dy = y - yc
        sigma_x2 = LMparams["sigma_x2"].value
        sigma_y2 = LMparams["sigma_y2"].value
        return a * math.exp(-(dx**2 / (2 * sigma_x2) + dy**2 / (2 * sigma_y2))) + b

    def psf_error(self, LMparams):
        subimage = self.subimage
        xs = [x for x in range(subimage.width) for y in range(subimage.height)]
        ys = [y for x in range(subimage.width) for y in range(subimage.height)]
        vals = [
            subimage.get_pixel(x, y)
            for x in range(subimage.width)
            for y in range(subimage.height)
        ]
        errors = [vals[i] - self.psf(LMparams, xs[i], ys[i]) for i in range(len(vals))]
        return errors


# Double Gaussian Model for Pluto Charon blob
# Estimates A_p, A_c, B, and the centers of Pluto and Charon
class PlutoCharonGaussian(GaussianModel):
    def __init__(self, PSFSetupData):
        super().__init__(PSFSetupData)
        self.LMparams = Parameters()
        self.LMparams.add("x_0p", value=self.PSFSetupData["x_0p"])
        self.LMparams.add("y_0p", value=self.PSFSetupData["y_0p"])
        self.LMparams.add("x_0c", value=self.PSFSetupData["x_0c"])
        self.LMparams.add("y_0c", value=self.PSFSetupData["y_0c"])
        self.LMparams.add("a_p", value=self.PSFSetupData["init_Ap"])
        self.LMparams.add("a_c", value=self.PSFSetupData["init_Ac"])
        self.LMparams.add("b", value=self.PSFSetupData["init_background"])

    def get_params(self):
        """
        Returns all seven params {x_0c, y_0c, x_0p,y_0p, A_p, A_c, B}
        """
        return super().get_params()

    def psf(self, LMparams, x, y):
        a_c = LMparams["a_c"].value
        a_p = LMparams["a_p"].value
        b = LMparams["b"].value
        sigma_x2 = self.PSFSetupData["sigma_x2"]
        sigma_y2 = self.PSFSetupData["sigma_y2"]
        dx_c = x - LMparams["x_0c"].value
        dy_c = y - LMparams["y_0c"].value
        dx_p = x - LMparams["x_0p"].value
        dy_p = y - LMparams["y_0p"].value
        return (
            a_c
            * math.exp(-((dx_c**2 / (2 * sigma_x2)) + (dy_c**2 / (2 * sigma_y2))))
            + a_p
            * math.exp(-((dx_p**2 / (2 * sigma_x2)) + (dy_p**2 / (2 * sigma_y2))))
            + b
        )

    def psf_error(self, LMparams):
        subimage = self.subimage
        xs = [x for x in range(subimage.width) for y in range(subimage.height)]
        ys = [y for x in range(subimage.width) for y in range(subimage.height)]
        vals = [
            subimage.get_pixel(x, y)
            for x in range(subimage.width)
            for y in range(subimage.height)
        ]
        errors = [vals[i] - self.psf(LMparams, xs[i], ys[i]) for i in range(len(vals))]
        return errors


def locate_pluto_charon(PlutoCharonSetupData):
    locations = {"x_p": [], "y_p": [], "x_c": [], "y_c": [], "pixel_distance": [], "arcsecond_distance": []}
    scale = 1
    dx_p = np.array([1, 1, -1, -1]) * scale
    dy_p = np.array([1, 1, 1, 1]) * scale
    dx_c = np.array([-1, 1, 1, -1]) * scale
    dy_c = np.array([-1, -1, -1, -1]) * scale
    # guess_x = PlutoCharonSetupData["subimage"].width/2
    # guess_y = PlutoCharonSetupData["subimage"].height/2
    guess_x = 9
    guess_y = 9
    print("Intial guesses", guess_x, guess_y)
    for i in range(len(dx_p)):  # test the four locations as per the convergence diagram
        PlutoCharonSetupData["x_0p"] = guess_x + dx_p[i]
        PlutoCharonSetupData["y_0p"] = guess_y + dy_p[i]
        PlutoCharonSetupData["x_0c"] = guess_x + dx_c[i]
        PlutoCharonSetupData["y_0c"] = guess_y + dy_c[i]
        pluto_charon = PlutoCharonGaussian(PlutoCharonSetupData)
        params = pluto_charon.get_params()
        print("Locations after fitting")
        print("Pluto: ", params["x_0p"].value, params["y_0p"].value)
        print("Charon: ", params["x_0c"].value, params["y_0c"].value)
        locations["x_p"].append(params["x_0p"].value)
        locations["y_p"].append(params["y_0p"].value)
        locations["x_c"].append(params["x_0c"].value)
        locations["y_c"].append(params["y_0c"].value)
        distance = math.sqrt((params["x_0p"].value - params["x_0c"].value)**2 + (params["y_0p"].value - params["y_0c"].value)**2)
        locations["pixel_distance"].append(distance)
        locations["arcsecond_distance"].append(distance * constants[config.date + config.index]["plate_scale"])
        if config.do_pauses_for_gaussian:
            input("Press enter to keep going")
    return locations


def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_params_from_file(file_path):
    fittings = Table.read(file_path, format="csv")
    return fittings


def plot_params(fittings, center_x, center_y):
    center_dist = []
    for x, y in zip(fittings["x"], fittings["y"]):  # iterate in parallel
        center_dist.append(math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
    print(center_dist, fittings["sigma_x2"])
    plt.title("Sigma_x2 as function from distance from center")
    plt.scatter(np.array(center_dist), np.array(fittings["sigma_x2"]), linestyle="None")
    plt.show()
    plt.figure()
    plt.title("Sigma_y2 as function from distance from center")
    plt.scatter(np.array(center_dist), np.array(fittings["sigma_y2"]), linestyle="None")
    plt.show()
    print("sigma_x2", str(np.average(fittings["sigma_x2"])))
    print("sigma_y2", str(np.average(fittings["sigma_y2"])))


def compute_avg(file_path, upper_bound=10):
    """compute the average sigma_x2 and sigma_y2. ignores vals over threshold"""
    fittings = get_params_from_file(file_path)

    avg_sigma_x2 = np.average([x for x in fittings["sigma_x2"] if x < upper_bound])
    avg_sigma_y2 = np.average([x for x in fittings["sigma_y2"] if x < upper_bound])
    return (avg_sigma_x2, avg_sigma_y2)
