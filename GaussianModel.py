"""
USAGE:
$ python3 GaussianModel.py date output_path
    date = MM-DD-YYYY date in which image was taken (this is the name of the data folder)
    output_path = path to output csv file to write to
or
$ python3 GaussianModel.py input_path
    - read from existing values
    input_path = path to csv file previously generated from above usage
"""

import math
import sys

from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from IStar import IStar
from Image import Image, get_image_hdu_number

error_threshold =  0.0 # maximum change in total residual error

def psf(params, x, y):
    return params["A"] * math.exp(-((x**2 / (2 * params["sigma_x2"])) + (y**2 / (2 * params["sigma_y2"])))) + params["B"]

def psf_error(params, xy, data, image, center_x, center_y):
    n = len(data)
    errors = []
    for i in range(n):
        dx = xy[0][i]
        dy = xy[1][i]
        errors.append(image.get_pixel(int(center_x - dx), int(center_y - dy)) - psf(params, dx, dy))
    return errors


class GaussianModel:
    def __init__(self, center_x, center_y, image, fwhm, average_pixel_value):
        self.center_x = center_x
        self.center_y = center_y
        self.image = image

        a = np.max(
            [ # four pixels around center
                image.get_pixel(math.floor(center_x), math.floor(center_y)),
                image.get_pixel(math.ceil(center_x), math.floor(center_y)),
                image.get_pixel(math.floor(center_x), math.ceil(center_y)),
                image.get_pixel(math.ceil(center_x), math.ceil(center_y)),
            ]
        )
        sigma2 = (fwhm/2.355)**2

        self.LMparams = Parameters()
        self.LMparams.add("A", value=a)
        self.LMparams.add("B", value=average_pixel_value)
        self.LMparams.add("sigma_x2", value=sigma2)
        self.LMparams.add("sigma_y2", value=sigma2)
        self.prev_residual_error = -math.inf

    def get_sigma(self):
        """
        Returns all four params {A,B,sigma_x2,sigma_y2} obtained from fitting
        """
        total_residual_error = 0
        dsda = dsdb = dsdsigma_x2 = dsdsigma_y2 = 0
        xs = []
        ys = []
        data = []

        a = self.LMparams["A"].value
        b = self.LMparams["B"].value
        sigma_x2 = self.LMparams["sigma_x2"].value
        sigma_y2 = self.LMparams["sigma_y2"].value

        for x in range(int(round(self.center_x)) - 10, int(round(self.center_x)) + 10):
            for y in range(int(round(self.center_y)) - 10, int(round(self.center_y)) + 10):
                dx = self.center_x - x # x = center_x - dx
                dy = self.center_y - y # y = center_y - dy
                val = psf(self.LMparams,dx, dy)

                xs.append(dx)
                ys.append(dy)
                data.append(val)

                g = self.image.get_pixel(x, y)
                exp_term = (val - b) / a

                total_residual_error += (self.image.get_pixel(x,y) - val) ** 2
                dsda += -2 * (g - val) * exp_term
                dsdb += -2 * (g - val) * 1
                dsdsigma_x2 += (
                    -2 * (g - val) * a * x ** 2 * (exp_term) / sigma_x2 ** 3
                )
                dsdsigma_y2 += (
                    -2 * (g - val) * a * y ** 2 *(exp_term) / sigma_y2 ** 3
                )
        # print(total_residual_error, self.prev_residual_error, total_residual_error - self.prev_residual_error)
        # print(dsda,dsdb,dsdsigma_x2,dsdsigma_y2)
        if abs(total_residual_error - self.prev_residual_error) <= error_threshold:
            return self.LMparams
        self.prev_residual_error = total_residual_error

        LMFitmin = Minimizer(psf_error, self.LMparams, fcn_args=([xs, ys], data, self.image, self.center_x, self.center_y))
        LMFitResult = LMFitmin.minimize(method="least_square")
        print(LMFitResult.params)
        self.LMparams = LMFitResult.params
        return self.get_sigma()

def distance(x1, x2, y1, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_sigmas_from_file(file_path):
    fittings = Table.read(file_path, format="csv")
    center_dist = []
    center_x, center_y = 635, 526 # need to parameterize
    for x, y in zip(fittings["x"], fittings["y"]): # iterate in parallel
        center_dist.append(math.sqrt((x-center_x)**2+(y-center_y)**2))
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


def main():
    n = len(sys.argv)
    if n == 2:
        fittings_path = "./out/Gaussian/" + sys.argv[1] +".csv"
        get_sigmas_from_file(fittings_path)
    else:
        path = "./data/" + sys.argv[1] + "/pluto_V.fits"
        starlist_path = "./out/Starlist/" + sys.argv[1] + ".csv"
        starlist = Table.read(starlist_path, format="csv")
        output_path = "./out/Gaussian/"+sys.argv[2] + ".csv"

        image = Image(path)
        hdul = fits.open(path)

        fwhm_arc = 3.5  # full width half maximum in arcseconds
        fwhm = (
            fwhm_arc / hdul[get_image_hdu_number(hdul)].header["CDELT1"]
        )  # fwhm in pixels

        all_params = {"star": [], "A": [], "B": [], "sigma_x2": [], "sigma_y2": [], "x": [], "y":[]}

        skip_count = 0

        for i in range(len(starlist)):
            star = IStar(table_row=starlist[i])
            print("<"+str(i+1)+">\n", str(starlist[i]))
            skip = False
            # ignore fake stars
            if not isinstance(star.counts, str):
                if star.counts < 0:
                    skip = True
            for j in range(len(starlist)):
                # ignore stars that are within 25 pixels of the current star to avoid PSF issuse
                star2 = IStar(table_row=starlist[j])
                if i!=j and distance(star.x, star2.x, star.y, star2.y) < 25:
                    skip = True
            if skip:
                print("SKIPPED")
                skip_count+=1
                continue
            gm = GaussianModel(star.x, star.y, image, fwhm, image.get_average_pixel_value())
            params = gm.get_sigma()
            # print(params)
            all_params["star"].append(star.star_name)
            all_params["A"].append(params["A"].value)
            all_params["B"].append(params["B"].value)
            all_params["sigma_x2"].append(params["sigma_x2"].value)
            all_params["sigma_y2"].append(params["sigma_y2"].value)
            all_params["x"].append(star.x)
            all_params["y"].append(star.y)

        print("Number of stars successfully analyzed:", len(starlist) - skip_count)
        print("sigma_x2", str(np.average(all_params["sigma_x2"])))
        print("sigma_y2", str(np.average(all_params["sigma_y2"])))

        Table(all_params).write(output_path, format="csv", overwrite=True)


if __name__ == "__main__":
    main()
