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
            -((x ** 2 / (2 * params["sigma_x2"])) + (y ** 2 / (2 * params["sigma_y2"])))
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


def pc_psf(params, x_p, y_p, x_c, y_c, sigma_x2, sigma_y2):
    return (
        params["a_c"]
        * math.exp(-((x_c ** 2 / (2 * sigma_x2)) + (y_c ** 2 / (2 * sigma_y2))))
        + params["a_p"]
        * math.exp(-((x_p ** 2 / (2 * sigma_x2)) + (y_p ** 2 / (2 * sigma_y2))))
        + params["B"]
    )


def pc_psf_error(params, pcxy, data, image, sigma_x2, sigma_y2):
    # pcxy is x_ps, y_ps, x_cs, y_cs
    n = len(data)
    errors = []
    for i in range(n):
        x_p = pcxy[0][i]
        y_p = pcxy[1][i]
        x_c = pcxy[2][i]
        y_c = pcxy[3][i]
        print("x_p", str(x_p))
        print("x_0p", str(params["x_0p"].value))
        print("y_p", str(y_p))
        print("y_0p", str(params["y_0p"].value))
        print(int(params["x_0p"].value - x_p), int(params["y_0p"].value - y_p))
        errors.append(
            image.get_pixel(
                int(params["x_0p"].value - x_p), int(params["y_0p"].value - y_p)
            )
            - pc_psf(params, x_p, y_p, x_c, y_c, sigma_x2, sigma_y2)
        )
    return errors


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
                dsdsigma_x2 += -2 * (g - val) * a * x ** 2 * (exp_term) / sigma_x2 ** 3
                dsdsigma_y2 += -2 * (g - val) * a * y ** 2 * (exp_term) / sigma_y2 ** 3
        # print(total_residual_error, self.prev_residual_error, total_residual_error - self.prev_residual_error)
        # print(dsda,dsdb,dsdsigma_x2,dsdsigma_y2)
        if abs(total_residual_error - self.prev_residual_error) <= error_threshold:
            return self.LMparams
        self.prev_residual_error = total_residual_error

        LMFitmin = Minimizer(
            psf_error,
            self.LMparams,
            fcn_args=([xs, ys], data, self.image, self.center_x, self.center_y),
        )
        LMFitResult = LMFitmin.minimize(method="least_square")
        print(LMFitResult.params)
        self.LMparams = LMFitResult.params
        return self.get_params()


class PlutoCharonGaussian:
    def __init__(
        self,
        image,
        a_p,
        a_c,
        b,
        sigma_x2,
        sigma_y2,
        pluto_x,
        pluto_y,
        charon_x,
        charon_y,
        center_x,
        center_y,
    ):
        self.sigma_x2 = sigma_x2
        self.sigma_y2 = sigma_y2
        self.image = image
        self.pluto_x = pluto_x
        self.pluto_y = pluto_y
        self.charon_x = charon_x
        self.charon_y = charon_y
        self.center_x = center_x  # center of blob identified by find stars
        self.center_y = center_y
        self.a_p = a_p
        self.a_c = a_c
        self.b = b

        self.LMparams = Parameters()
        self.LMparams.add("x_0p", value=pluto_x)
        self.LMparams.add("y_0p", value=pluto_y)
        self.LMparams.add("x_0c", value=charon_x)
        self.LMparams.add("y_0c", value=charon_y)
        self.LMparams.add("a_p", value=a_p)
        self.LMparams.add("a_c", value=a_c)
        self.LMparams.add("B", value=b)
        self.prev_residual_error = -math.inf

    def get_params(self):
        """
        Returns all seven params {x_0c, y_0c, x_0p,y_0p, A_p, A_c, B}
        """
        total_residual_error = 0
        x_ps = []
        y_ps = []
        x_cs = []
        y_cs = []
        data = []
        sigma_x2 = self.sigma_x2
        sigma_y2 = self.sigma_y2
        x_0p = self.LMparams["x_0p"].value
        y_0p = self.LMparams["y_0p"].value
        x_0c = self.LMparams["x_0c"].value
        y_0c = self.LMparams["y_0c"].value
        # call psf on 20x20 box around center of pluto charon blob
        for x in range(int(round(self.center_x)) - 10, int(round(self.center_x)) + 10):
            for y in range(
                int(round(self.center_y)) - 10, int(round(self.center_y)) + 10
            ):
                x_p = x_0p - x
                y_p = y_0p - y
                x_c = x_0c - x
                y_c = y_0c - y
                val = pc_psf(self.LMparams, x_p, y_p, x_c, y_c, sigma_x2, sigma_y2)

                x_ps.append(x_p)
                y_ps.append(y_p)
                x_cs.append(x_c)
                y_cs.append(y_c)
                data.append(val)

                total_residual_error += (self.image.get_pixel(x, y) - val) ** 2
        # keep going until no residual error
        if abs(total_residual_error - self.prev_residual_error) <= error_threshold:
            return self.LMparams
        self.prev_residual_error = total_residual_error

        LMFitmin = Minimizer(
            pc_psf_error,
            self.LMparams,
            fcn_args=(
                [x_ps, y_ps, x_cs, y_cs],
                data,
                self.image,
                self.sigma_x2,
                self.sigma_y2,
            ),
        )
        print("Before")
        LMFitResult = LMFitmin.minimize(method="least_square")  # code breaks here!
        print("After")
        print(LMFitResult.params)
        self.LMparams = LMFitResult.params
        return self.get_params()


def locate_pluto_charon(image, counts, center_x, center_y, sigma_x2, sigma_y2, b):
    a_p = 5 / 6 * counts # guess as 5/6 the brightness of pluto charon blob
    a_c = a_p / 5 # 1/5 of a_p
    scale = 2
    dx_p = np.array([1, 1, -1, -1]) * scale
    dy_p = np.array([1, -1, -1, 1]) * scale
    dx_c = np.array([-1, 1, 1, -1]) * scale
    dy_c = np.array([-1, -1, -1, -1]) * scale
    for i in range(len(dx_p)): # test the four locations as per the convergence diagram
        pluto_charon = PlutoCharonGaussian(
            image=image,
            b=b,
            a_p=a_p,
            a_c=a_c,
            sigma_x2=sigma_x2,
            sigma_y2=sigma_y2,
            pluto_x=center_x + dx_p[i],
            pluto_y=center_y + dy_p[i],
            charon_x=center_x + dx_c[i],
            charon_y=center_y + dy_c[i],
            center_x=center_x,
            center_y=center_y,
        )
        params = pluto_charon.get_params()

        print(params["x_0p"].value, params["y_0p"].value)
        print(params["x_0c"].value, params["y_0c"].value)


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


def main():
    # 4-25-2021
    print("Start")
    counts = 22790.0
    center_x = 636.87
    center_y = 555.8
    sigma_x2 = 10.558393348078292  # average from file
    sigma_y2 = 5.177641522106213  # average from file
    image = Image("./data/4-25-2021/pluto_V.fits")
    b = 4000  # background brightness
    locate_pluto_charon(image, counts, center_x, center_y, sigma_x2, sigma_y2, b)


def main2():
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

        print("Number of stars successfully analyzed:", len(starlist) - skip_count)
        print("sigma_x2", str(np.average(all_params["sigma_x2"])))
        print("sigma_y2", str(np.average(all_params["sigma_y2"])))

        Table(all_params).write(output_path, format="csv", overwrite=True)


if __name__ == "__main__":
    main()
