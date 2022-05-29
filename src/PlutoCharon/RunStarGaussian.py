import time
import glob

from GaussianModel import *
from Image import *
from res import config
from res.constants import *

def main():
    files = glob.glob(config.data_folder + config.date + "/*.fits").sort()
    path = files[0]
    starlist_path = config.data_folder + config.date + "/starlist.csv"

    starlist = Table.read(starlist_path, format="csv")
    output_path = (
        config.output_folder + config.date + "/" + time.strftime("%m-%d-%y") + ".csv"
    )

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

        # filter out bad stars
        skip = False
        if not isinstance(star.counts, str):  # if counts not "N/A"
            if star.counts < 0:
                skip = True
        for j in range(len(starlist)):
            # ignore stars that are within ... pixels of the current star to avoid interference
            star2 = IStar(table_row=starlist[j])
            if (
                i != j
                and distance(star.x, star2.x, star.y, star2.y) < config.min_distance
            ):
                skip = True
        # stars that are too close to the border
        if star.x < 11 or star.x > image.width - 11:
            skip = True
        if star.y < 11 or star.y > image.height - 11:
            skip = True
        if skip:
            if config.do_debugging_for_gaussian:
                print("==================SKIPPED================")
            skip_count += 1
            continue

        if config.do_debugging_for_gaussian:
            print("<" + str(i + 1) + ">\n", str(starlist[i]))
        PSFSetupData = {}
        PSFSetupData["star_x"] = star.x
        PSFSetupData["star_y"] = star.y
        PSFSetupData["orig_image"] = image
        PSFSetupData["subimage"] = PSFSetupData["orig_image"].subimage(
            star.x + 1,
            star.y + 1,
            19,
            19,
        )
        PSFSetupData["fwhm"] = fwhm
        PSFSetupData["avg_pixel_val"] = image.get_average_pixel_value()

        gm = StarGaussian(PSFSetupData)

        params = gm.get_params()
        all_params["star"].append(star.star_name)
        all_params["a"].append(params["a"].value)
        all_params["b"].append(params["b"].value)
        all_params["sigma_x2"].append(params["sigma_x2"].value)
        all_params["sigma_y2"].append(params["sigma_y2"].value)
        all_params["x"].append(star.x)
        all_params["y"].append(star.y)

        if config.do_pauses_for_gaussian:
            input("Press enter to keep going")

    print("\n\n" + "SUMMARY")
    print("Number of stars successfully analyzed:", len(starlist) - skip_count)
    print("sigma_x2", str(np.average([x for x in all_params["sigma_x2"] if x < 10])))
    print("sigma_y2", str(np.average([y for y in all_params["sigma_y2"] if y < 10])))

    Table(all_params).write(output_path, format="csv", overwrite=True)
    print("Wrote to file: " + output_path)

if __name__ == "__main__":
    main()