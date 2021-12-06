# https://witeboard.com/0bbe7650-4839-11ec-87f9-5f988fea8bbc

"""
ignore this for now
USAGE:
$ python3 GaussianModel.py date output_path
    date = MM-DD-YYYY date in which image was taken (this is the name of the data folder)
    output_path = path to output csv file to write to
or
$ python3 GaussianModel.py input_path
    - read from existing values
    input_path = path to csv file previously generated from above usage
"""

"""
two main files are:
PlutoCharonDriver
GeneralPSFDriver
"""

import math
import sys

from lmfit import Minimizer, minimize, Parameters, Parameter, fit_report
from scipy import optimize
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from IStar import IStar
from Image import Image, get_image_hdu_number


class GaussianModel:
    def __init__(self, PSFSetupData):
        self.PSFSetupData = PSFSetupData

    def get_params(self):
        subimage = self.PSFSetupData["subimage"]
        xc = int(subimage.width / 2 + 1)
        yc = int(subimage.height / 2 + 1)
        xs = [x - xc for x in range(subimage.width) for y in range(subimage.height)]
        ys = [y - yc for x in range(subimage.width) for y in range(subimage.height)]
        vals = [
            subimage.get_pixel(x, y)
            for x in range(subimage.width)
            for y in range(subimage.height)
        ]
        LMFitResult = minimize(
            self.psf_error,
            self.LMparams,
            args=(
                xs,
                ys,
                vals,
            ),
            method="least_squares",
        )
        # print(LMFitResult.params)
        # print(fit_report(LMFitResult))
        self.LMparams = LMFitResult.params
        return self.LMparams

    def psf(self):
        pass

    def psf_error(self):
        pass


class StarGaussian(GaussianModel):
    def __init__(self, PSFSetupData):
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
        self.LMparams.add("a", value=a)
        self.LMparams.add("b", value=PSFSetupData["avg_pixel_val"])
        self.LMparams.add("sigma_x2", value=sigma2)
        self.LMparams.add("sigma_y2", value=sigma2)
        super().__init__(PSFSetupData)

    def get_params(self):
        return super().get_params()

    def psf(self, LMparams, dx, dy):
        a = LMparams["a"].value
        b = LMparams["b"].value
        sigma_x2 = LMparams["sigma_x2"].value
        sigma_y2 = LMparams["sigma_y2"].value
        return a * math.exp(-(dx ** 2 / (2 * sigma_x2) + dy ** 2 / (2 * sigma_y2))) + b

    def psf_error(self, LMparams, xs, ys, vals):
        errors = [vals[i] - self.psf(LMparams, xs[i], ys[i]) for i in range(len(vals))]
        return errors


class PlutoCharonGaussian(GaussianModel):
    def __init__(self, PlutoCharonSetupData):
        self.LMparams = Parameters()
        self.LMparams.add("x_0p", value=self.PlutoCharonSetupData["x_0p"])
        self.LMparams.add("y_0p", value=self.PlutoCharonSetupData["y_0p"])
        self.LMparams.add("x_0c", value=self.PlutoCharonSetupData["x_0c"])
        self.LMparams.add("y_0c", value=self.PlutoCharonSetupData["y_0c"])
        self.LMparams.add("a_p", value=self.PlutoCharonSetupData["init_Ap"])
        self.LMparams.add("a_c", value=self.PlutoCharonSetupData["init_Ac"])
        self.LMparams.add("b", value=self.PlutoCharonSetupData["init_background"])

    def get_params(self):
        """
        Returns all seven params {x_0c, y_0c, x_0p,y_0p, A_p, A_c, B}
        """
        return super().get_params()

    def psf(self, LMparams, dx, dy):
        a_c = LMparams["a_c"].value
        a_p = LMparams["a_p"].value
        b = LMparams["b"].value
        sigma_x2 = self.PlutoCharonSetupData["sigma_x2"]
        sigma_y2 = self.PlutoCharonSetupData["sigma_y2"]
        # center of pluto + displacement relative to pluto = center of subimage + displacement relative to center of subimage
        # ==> displacement relative to pluto = center of subimage + displacement relative to subimage - center of pluto
        x_c = 10 + dx - (10 + LMparams["x_0c"].value)  # dx offset from pluto center
        y_c = dy - LMparams["y_0c"].value
        x_p = dx - LMparams["x_0p"].value
        y_p = dy - LMparams["y_0p"].value
        return (
            a_c * math.exp(-((x_c ** 2 / (2 * sigma_x2)) + (y_c ** 2 / (2 * sigma_y2))))
            + a_p
            * math.exp(-((x_p ** 2 / (2 * sigma_x2)) + (y_p ** 2 / (2 * sigma_y2))))
            + b
        )

    def psf_error(self, LMparams, xs, ys, vals):
        errors = [vals[i] - self.psf(LMparams, xs[i], ys[i]) for i in range(len(vals))]
        return errors


def locate_pluto_charon(PlutoCharonSetupData):
    scale = 1
    dx_p = np.array([1, 1, -1, -1]) * scale
    dy_p = np.array([1, -1, -1, 1]) * scale
    dx_c = np.array([-1, 1, 1, -1]) * scale
    dy_c = np.array([-1, -1, -1, -1]) * scale
    for i in range(len(dx_p)):  # test the four locations as per the convergence diagram
        # These coordinates are relative to the center of the subimage
        PlutoCharonSetupData["x_0p"] = 10 + dx_p[i]
        # CHANGE 10 TO MIDDLE OF SUBIMAGE LATER
        PlutoCharonSetupData["y_0p"] = 10 + dy_p[i]
        PlutoCharonSetupData["x_0c"] = 10 + dx_c[i]
        PlutoCharonSetupData["y_0c"] = 10 + dy_c[i]
        pluto_charon = PlutoCharonGaussian(PlutoCharonSetupData)
        params = pluto_charon.get_params()
        print("Locations after fitting")
        print("Pluto: ", params["x_0p"].value, params["y_0p"].value)
        print("Charon: ", params["x_0c"].value, params["y_0c"].value)


def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_params_from_file(file_path):
    fittings = Table.read(file_path, format="csv")
    # center_x, center_y = 635, 526  # need to parameterize center of image
    # plot_params(fittings, center_x, center_y)
    # print average sigma
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
