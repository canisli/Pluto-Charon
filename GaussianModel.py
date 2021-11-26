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
def psf(params, PSFSetupData, x, y):
    a = params["a"].value
    b = params["b"].value
    sigma_x2 = params["sigma_x2"].value
    sigma_y2 = params["sigma_y2"].value
    return (
        a * math.exp(-(x ** 2 / (2 * sigma_x2) + y ** 2 / (2 * sigma_y2))) + b
    )


def psf_error(params, PSFSetupData, xs, ys, vals):
    # return array of residuals
    errors = []
    for i in range(len(vals)):
        errors.append(vals[i]-psf(params, PSFSetupData, xs[i], ys[i]))
    return errors


def pc_psf(params, PlutoCharonSetupData, x, y):
    # x and y are displacements from the center of the image
    a_c = params["a_c"].value
    a_p = params["a_p"].value
    b = params["b"].value
    sigma_x2 = PlutoCharonSetupData["sigma_x2"]
    sigma_y2 = PlutoCharonSetupData["sigma_y2"]
    # center of pluto + displacement relative to pluto = center of subimage + displacement relative to center of subimage
    # ==> displacement relative to pluto = center of subimage + displacement relative to subimage - center of pluto
    x_c = 10+x-(10+params["x_0c"].value)  # x offset from pluto center
    y_c = y - params["y_0c"].value
    x_p = x - params["x_0p"].value
    y_p = y - params["y_0p"].value
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
    def __init__(self, PSFSetupData):
        self.PSFSetupData = PSFSetupData
        center_x = PSFSetupData["star_x"]
        center_y = PSFSetupData["star_y"]
        image = PSFSetupData["orig_image"]
        fwhm = PSFSetupData["fwhm"]
        avg_pixel_val = PSFSetupData["avg_pixel_val"]

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
        self.LMparams.add("a", value=a)
        self.LMparams.add("b", value=avg_pixel_val)
        self.LMparams.add("sigma_x2", value=sigma2)
        self.LMparams.add("sigma_y2", value=sigma2)
        # self.prev_residual_error = -math.inf

    def get_params(self):
        """
        Returns all four params {A,B,sigma_x2,sigma_y2} obtained from fitting
        """
        # total_residual_error = 0
        #dsda = dsdb = dsdsigma_x2 = dsdsigma_y2 = 0

        center_x = self.PSFSetupData["star_x"]
        center_y = self.PSFSetupData["star_y"]
        subimage = self.PSFSetupData["subimage"]

        xs = []  # make into dxs
        ys = []
        vals = []  # actual pixel value
        # call psf on 20x20 box around center of pluto charon blob
        for x in range(0, 19):  # TODO de-hardcode it
            for y in range(0, 19):
                xs.append(x-10)
                # TODO MAKE COORDINATES RELATIVE TO THE CENTER OF THE STAR
                ys.append(y-10)
                vals.append(self.PSFSetupData["subimage"].get_pixel(x, y))

        # LMFitmin = Minimizer(
        #     psf_error,
        #     self.LMparams,
        #     args=(xs, ys, vals,)
        # )
        LMFitResult = minimize(psf_error, self.LMparams, args=(
            self.PSFSetupData, xs, ys, vals,), method="least_squares")
        # print(LMFitResult.params)
        # print(fit_report(LMFitResult))
        self.LMparams = LMFitResult.params
        return self.LMparams


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
        # call psf on 19x19 box around center of pluto charon blob
        for x in range(0, 19):  # TODO de-hardcode it
            for y in range(0, 19):
                # TODO MAKE COORDINATES RELATIVE TO THE CENTER OF THE STAR
                xs.append(x-10)
                ys.append(x-10)
                vals.append(
                    self.PlutoCharonSetupData["subimage"].get_pixel(x, y))

        # print(len(xs), len(ys), len(vals))
        # print(vals)
        # LMFitmin = Minimizer(
        #     pc_psf_error,
        #     self.LMparams,
        #     fcn_args=(
        #         self.PlutoCharonSetupData,
        #         xs, ys, vals
        #     ),
        # )
        LMFitResult = minimize(pc_psf_error, self.LMparams, args=(
            self.PlutoCharonSetupData, xs, ys, vals,), method="least_squares")
        # print("Before")
        # print("After")
        print(fit_report(LMFitResult))
        return LMFitResult.params


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
    # center_x, center_y = 635, 526  # need to parameterize center of image
    #plot_params(fittings, center_x, center_y)
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


def compute_avg(file_path, upper_bound=10):
    """compute the average sigma_x2 and sigma_y2. ignores vals over threshold"""
    fittings = get_params_from_file(file_path)

    avg_sigma_x2 = np.average(
        [x for x in fittings["sigma_x2"] if x < upper_bound])
    avg_sigma_y2 = np.average(
        [x for x in fittings["sigma_y2"] if x < upper_bound])
    return (avg_sigma_x2, avg_sigma_y2)


def PlutoCharonDriver():
    # 4-25-2021
    print("Start")
    counts = 22790.0  # counts of unidentified star (Pluto and Charon)
    PlutoCharonSetupData = {}
    PlutoCharonSetupData["orig_image"] = Image("./data/4-25-2021/pluto_V.fits")
    blob_center_x = 636.87+2
    blob_center_y = 555.8 + 2
    PlutoCharonSetupData["subimage"] = PlutoCharonSetupData["orig_image"].subimage(
        blob_center_x, blob_center_y, 19, 19
    )
    # estimate based off grabbing values from ds9
    PlutoCharonSetupData["init_background"] = 4000
    PlutoCharonSetupData["init_Ap"] = 5 / 6 * counts  # guess
    PlutoCharonSetupData["init_Ac"] = 1 / 6 * counts
    PlutoCharonSetupData["blob_center_x"] = blob_center_x
    PlutoCharonSetupData["blob_center_y"] = blob_center_y
    # average from GaussianModel.get_params
    avg_sigmas = compute_avg("./out/Gaussian/11-25-2021.csv")
    PlutoCharonSetupData["sigma_x2"] = avg_sigmas[0]
    PlutoCharonSetupData["sigma_y2"] = avg_sigmas[1]
    PlutoCharonSetupData["subimage"].write_fits("4-25-2021_PC_subimage")
    print(PlutoCharonSetupData["subimage"].get_pixel(5, 5))
    locate_pluto_charon(PlutoCharonSetupData)


def GeneralPSFDriver():  # for general PSF Gaussian
    n = len(sys.argv)
    if n == 2:
        fittings_path = "./out/Gaussian/" + sys.argv[1] + ".csv"
        get_params_from_file(fittings_path)
    else:
        # path = "./data/" + sys.argv[1] + "/pluto_V.fits"
        # starlist_path = "./out/Starlist/" + sys.argv[1] + ".csv"
        # starlist = Table.read(starlist_path, format="csv")
        # output_path = "./out/Gaussian/" + sys.argv[2] + ".csv"
        path = "./data/4-25-2021/pluto_V.fits"
        starlist_path = "./out/Starlist/4-25-2021.csv"
        starlist = Table.read(starlist_path, format="csv")
        output_path = "./out/Gaussian/11-25-2021.csv"

        image = Image(path)
        hdul = fits.open(path)

        fwhm_arc = 3.5  # full width half maximum in arcseconds
        fwhm = (
            fwhm_arc / hdul[get_image_hdu_number(hdul)].header["CDELT1"]
        )  # fwhm in pixels

        all_params = {  # dict for all the stars
            "star": [],
            "a": [],
            "b": [],
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
            if not isinstance(star.counts, str):  # if counts not "N/A"
                if star.counts < 0:
                    skip = True
            for j in range(len(starlist)):
                # ignore stars that are within 25 pixels of the current star to avoid PSF issuse
                star2 = IStar(table_row=starlist[j])
                if i != j and distance(star.x, star2.x, star.y, star2.y) < 25:
                    skip = True
            if skip:
                print("==================SKIPPED==============")
                skip_count += 1
                continue

            PSFSetupData = {}
            PSFSetupData["star_x"] = star.x
            PSFSetupData["star_y"] = star.y
            PSFSetupData["orig_image"] = image
            PSFSetupData["subimage"] = PSFSetupData["orig_image"].subimage(
                star.x, star.y, 19, 19
            )
            PSFSetupData["subimage"].write_fits("nov25")
            PSFSetupData["fwhm"] = fwhm
            PSFSetupData["avg_pixel_val"] = image.get_average_pixel_value()

            gm = GaussianModel(PSFSetupData)

            params = gm.get_params()
            # print(params)
            all_params["star"].append(star.star_name)
            all_params["a"].append(params["a"].value)
            all_params["b"].append(params["b"].value)
            all_params["sigma_x2"].append(params["sigma_x2"].value)
            all_params["sigma_y2"].append(params["sigma_y2"].value)
            all_params["x"].append(star.x)
            all_params["y"].append(star.y)

        print("Number of stars successfully analyzed:",
              len(starlist) - skip_count)
        print("sigma_x2", str(np.average(all_params["sigma_x2"])))
        print("sigma_y2", str(np.average(all_params["sigma_y2"])))

        Table(all_params).write(output_path, format="csv", overwrite=True)
        print("Wrote to file: " + output_path)


def main():
    PlutoCharonDriver()


if __name__ == "__main__":
    main()
