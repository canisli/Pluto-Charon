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

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from scipy import optimize
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from IStar import IStar
from Image import Image, get_image_hdu_number

error_threshold = 0.0  # maximum change in total residual error


def psf(params, x, y):
    return (
        params["A"]
        * math.exp(
            -((x ** 2 / (2 * params["sigma_x2"].value)) +
              (y ** 2 / (2 * params["sigma_y2"].value)))
        )
        + params["B"]
    )


def psf_error(params, xy, data, image, center_x, center_y):
    n = len(data)
    errors = []
    for i in range(n):
        dx = xy[0][i]
        dy = xy[1][i]
        errors.append(
            image.get_pixel(int(center_x - dx), int(center_y - dy))
            - psf(params, dx, dy)
        )
    return errors


def pc_psf(params, PlutoCharonSetupData, x, y):
    # x and y are displacements from the center of the image
    a_c = params["a_c"].value
    a_p = params["a_p"].value
    b = params["b"].value
    sigma_x2 = PlutoCharonSetupData["sigma_x2"]
    sigma_y2 = PlutoCharonSetupData["sigma_y2"]
    x_c = params["x_0c"].value - x
    y_c = params["y_0c"].value - y
    x_p = params["x_0p"].value - x
    y_p = params["y_0p"].value - y
    return (
        a_c * math.exp(-((x_c ** 2 / (2 * sigma_x2)) + (y_c ** 2 / (2 * sigma_y2)))) +
        a_p * math.exp(-((x_p ** 2 / (2 * sigma_x2)) +
                       (y_p ** 2 / (2 * sigma_y2))))
        + b
    )


def pc_psf_error(params, PlutoCharonSetupData, xs, ys, vals):
    errors = []
    for i in range(len(vals)):
        errors.append(
            vals[i] - pc_psf(params, PlutoCharonSetupData, xs[i], ys[i]))
    return errors
    # return val - pc_psf(params, PlutoCharonSetupData, x, y)


class GaussianModel:
    def __init__(self, center_x, center_y, image, fwhm, average_pixel_value):
        self.center_x = center_x
        self.center_y = center_y
        self.image = image

        a = np.max(
            [  # four pixels around center
                image.get_pixel(math.floor(center_x), math.floor(center_y)),
                image.get_pixel(math.ceil(center_x), math.floor(center_y)),
                image.get_pixel(math.floor(center_x), math.ceil(center_y)),
                image.get_pixel(math.ceil(center_x), math.ceil(center_y)),
            ]
        )
        sigma2 = (fwhm / 2.355) ** 2

        self.LMparams = Parameters()
        self.LMparams.add("A", value=a)
        self.LMparams.add("B", value=average_pixel_value)
        self.LMparams.add("sigma_x2", value=sigma2)
        self.LMparams.add("sigma_y2", value=sigma2)
        self.prev_residual_error = -math.inf

    def get_params(self):
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
            for y in range(
                int(round(self.center_y)) - 10, int(round(self.center_y)) + 10
            ):
                dx = self.center_x - x  # x = center_x - dx
                dy = self.center_y - y  # y = center_y - dy
                val = psf(self.LMparams, dx, dy)

                xs.append(dx)
                ys.append(dy)
                data.append(val)

                g = self.image.get_pixel(x, y)
                exp_term = (val - b) / a

                total_residual_error += (self.image.get_pixel(x, y) - val) ** 2
                dsda += -2 * (g - val) * exp_term
                dsdb += -2 * (g - val) * 1
                dsdsigma_x2 += -2 * (g - val) * a * x ** 2 * \
                    (exp_term) / sigma_x2 ** 3
                dsdsigma_y2 += -2 * (g - val) * a * y ** 2 * \
                    (exp_term) / sigma_y2 ** 3
        # print(total_residual_error, self.prev_residual_error, total_residual_error - self.prev_residual_error)
        # print(dsda,dsdb,dsdsigma_x2,dsdsigma_y2)
        if abs(total_residual_error - self.prev_residual_error) <= error_threshold:
            return self.LMparams
        self.prev_residual_error = total_residual_error

        LMFitmin = Minimizer(
            psf_error,
            self.LMparams,
            fcn_args=([xs, ys], data, self.image,
                      self.center_x, self.center_y),
        )
        LMFitResult = LMFitmin.minimize(method="least_square")
        print(LMFitResult.params)
        self.LMparams = LMFitResult.params
        return self.get_params()


class PlutoCharonGaussian:
    def __init__(self, PlutoCharonSetupData):
        self.PlutoCharonSetupData = PlutoCharonSetupData

        self.LMparams = Parameters()
        self.LMparams.add("x_0p", value=self.PlutoCharonSetupData["x_0p"])
        self.LMparams.add("y_0p", value=self.PlutoCharonSetupData["y_0p"])
        self.LMparams.add("x_0c", value=self.PlutoCharonSetupData["x_0c"])
        self.LMparams.add("y_0c", value=self.PlutoCharonSetupData["y_0c"])
        self.LMparams.add("a_p", value=self.PlutoCharonSetupData["init_Ap"])
        self.LMparams.add("a_c", value=self.PlutoCharonSetupData["init_Ac"])
        self.LMparams.add(
            "b", value=self.PlutoCharonSetupData["init_background"])

    def get_params(self):
        """
        Returns all seven params {x_0c, y_0c, x_0p,y_0p, A_p, A_c, B}
        """
        xs = []
        ys = []
        vals = []  # actualy pixel value
        # call psf on 20x20 box around center of pluto charon blob
        for x in range(0, 19):  # TODO de-hardcode it
            for y in range(0, 19):
                xs.append(x)
                ys.append(y),
                vals.append(
                    self.PlutoCharonSetupData["subimage"].get_pixel(x, y))

        # print(len(xs), len(ys), len(vals))
        # print(vals)
        LMFitmin = Minimizer(
            pc_psf_error,
            self.LMparams,
            fcn_args=(
                self.PlutoCharonSetupData,
                xs, ys, vals
            ),
        )
        # print("Before")
        LMFitResult = LMFitmin.minimize(
            method="least_squares")  # code breaks here!
        # print("After")
        # print(LMFitResult.params)
        return LMFitResult.params


def locate_pluto_charon(PlutoCharonSetupData):
    scale = 2
    dx_p = np.array([1, 1, -1, -1]) * scale
    dy_p = np.array([1, -1, -1, 1]) * scale
    dx_c = np.array([-1, 1, 1, -1]) * scale
    dy_c = np.array([-1, -1, -1, -1]) * scale
    for i in range(len(dx_p)):  # test the four locations as per the convergence diagram
        PlutoCharonSetupData["x_0p"] = 10 + dx_p[i]
        # CHANGE 10 TO MIDDLE LATER
        PlutoCharonSetupData["y_0p"] = 10 + dy_p[i]
        PlutoCharonSetupData["x_0c"] = 10 + dx_c[i]
        PlutoCharonSetupData["y_0c"] = 10 + dy_c[i]
        pluto_charon = PlutoCharonGaussian(
            PlutoCharonSetupData
        )
        params = pluto_charon.get_params()
        print("Locations after fitting")
        print("Pluto: ", params["x_0p"].value, params["y_0p"].value)
        print("Charon: ", params["x_0c"].value, params["y_0c"].value)


def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_params_from_file(file_path):
    fittings = Table.read(file_path, format="csv")
    center_x, center_y = 635, 526  # need to parameterize
    plot_params(fittings, center_x, center_y)
    # print average sigma
    return fittings


def plot_params(fittings, center_x, center_y):
    center_dist = []
    for x, y in zip(fittings["x"], fittings["y"]):  # iterate in parallel
        center_dist.append(
            math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
    print(center_dist, fittings["sigma_x2"])
    plt.title("Sigma_x2 as function from distance from center")
    plt.scatter(np.array(center_dist), np.array(
        fittings["sigma_x2"]), linestyle="None")
    plt.show()
    plt.figure()
    plt.title("Sigma_y2 as function from distance from center")
    plt.scatter(np.array(center_dist), np.array(
        fittings["sigma_y2"]), linestyle="None")
    plt.show()
    print("sigma_x2", str(np.average(fittings["sigma_x2"])))
    print("sigma_y2", str(np.average(fittings["sigma_y2"])))


def main():
    # 4-25-2021
    print("Start")
    counts = 22790.0  # counts of unidentified star (Pluto and Charon)

    PlutoCharonSetupData = {}
    PlutoCharonSetupData["orig_image"] = Image("./data/4-25-2021/pluto_V.fits")
    PlutoCharonSetupData["subimage"] = PlutoCharonSetupData["orig_image"].subimage(
        636.87 + 2, 555.8 + 2, 19, 19
    )
    # estimate based off grabbing values from ds9
    PlutoCharonSetupData["init_background"] = 4000
    PlutoCharonSetupData["init_Ap"] = 5 / 6 * counts  # guess
    PlutoCharonSetupData["init_Ac"] = 1 / 6 * counts
    PlutoCharonSetupData["blob_center_x"] = 636.87
    PlutoCharonSetupData["blob_center_y"] = 555.8
    # average from GaussianModel.get_params
    PlutoCharonSetupData["sigma_x2"] = 10.558393348078292
    PlutoCharonSetupData["sigma_y2"] = 5.177641522106213
    PlutoCharonSetupData["subimage"].write_fits("4-25-2021_PC_subimage")
    print(PlutoCharonSetupData["subimage"].get_pixel(5, 5))
    locate_pluto_charon(PlutoCharonSetupData)


def main2():  # for general PSF Gaussian
    n = len(sys.argv)
    if n == 2:
        fittings_path = "./out/Gaussian/" + sys.argv[1] + ".csv"
        get_params_from_file(fittings_path)
    else:
        path = "./data/" + sys.argv[1] + "/pluto_V.fits"
        starlist_path = "./out/Starlist/" + sys.argv[1] + ".csv"
        starlist = Table.read(starlist_path, format="csv")
        output_path = "./out/Gaussian/" + sys.argv[2] + ".csv"

        image = Image(path)
        hdul = fits.open(path)

        fwhm_arc = 3.5  # full width half maximum in arcseconds
        fwhm = (
            fwhm_arc / hdul[get_image_hdu_number(hdul)].header["CDELT1"]
        )  # fwhm in pixels

        all_params = {
            "star": [],
            "A": [],
            "B": [],
            "sigma_x2": [],
            "sigma_y2": [],
            "x": [],
            "y": [],
        }

        skip_count = 0

        for i in range(len(starlist)):
            star = IStar(table_row=starlist[i])
            print("<" + str(i + 1) + ">\n", str(starlist[i]))
            skip = False
            # ignore fake stars
            if not isinstance(star.counts, str):
                if star.counts < 0:
                    skip = True
            for j in range(len(starlist)):
                # ignore stars that are within 25 pixels of the current star to avoid PSF issuse
                star2 = IStar(table_row=starlist[j])
                if i != j and distance(star.x, star2.x, star.y, star2.y) < 25:
                    skip = True
            if skip:
                print("SKIPPED")
                skip_count += 1
                continue
            gm = GaussianModel(
                star.x, star.y, image, fwhm, image.get_average_pixel_value()
            )
            params = gm.get_params()
            # print(params)
            all_params["star"].append(star.star_name)
            all_params["A"].append(params["A"].value)
            all_params["B"].append(params["B"].value)
            all_params["sigma_x2"].append(params["sigma_x2"].value)
            all_params["sigma_y2"].append(params["sigma_y2"].value)
            all_params["x"].append(star.x)
            all_params["y"].append(star.y)

        print("Number of stars successfully analyzed:",
              len(starlist) - skip_count)
        print("sigma_x2", str(np.average(all_params["sigma_x2"])))
        print("sigma_y2", str(np.average(all_params["sigma_y2"])))

        Table(all_params).write(output_path, format="csv", overwrite=True)


if __name__ == "__main__":
    main()
