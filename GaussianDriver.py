import time
from GaussianModel import *


def PlutoCharonDriver():
    # 4-25-2021
    print("Start")
    counts = 22790.0  # counts of unidentified star (Pluto and Charon)
    PlutoCharonSetupData = {}
    PlutoCharonSetupData["orig_image"] = Image("./data/4-25-2021/pluto_V.fits")
    blob_center_x = 636.87 + 2
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
        output_path = "./out/Gaussian/" + time.strftime("%m-%d-%y") + ".csv"

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
            # print("<" + str(i + 1) + ">\n", str(starlist[i]))
            skip = False
            # ignore fake stars
            if not isinstance(star.counts, str):  # if counts not "N/A"
                if star.counts < 0:
                    skip = True
            for j in range(len(starlist)):
                # ignore stars that are within 25 pixels of the current star to avoid interference
                star2 = IStar(table_row=starlist[j])
                if i != j and distance(star.x, star2.x, star.y, star2.y) < 50:
                    skip = True
            if star.x < 11 or star.x > image.width - 11:
                skip = True
            if star.y < 11 or star.y > image.height - 11:
                skip = True
            if skip:
                print("==================SKIPPED==============")
                skip_count += 1
                continue
            print("<" + str(i + 1) + ">\n", str(starlist[i]))
            PSFSetupData = {}
            PSFSetupData["star_x"] = star.x
            PSFSetupData["star_y"] = star.y
            PSFSetupData["orig_image"] = image
            PSFSetupData["subimage"] = PSFSetupData["orig_image"].subimage(
                star.x + 1, star.y + 1, 19, 19 ########################################## ADD 1 TO BOTH
            )
            # PSFSetupData["subimage"].write_fits("nov25")
            PSFSetupData["fwhm"] = fwhm
            PSFSetupData["avg_pixel_val"] = image.get_average_pixel_value()

            gm = StarGaussian(PSFSetupData)

            params = gm.get_params()
            # print(params)
            all_params["star"].append(star.star_name)
            all_params["a"].append(params["a"].value)
            all_params["b"].append(params["b"].value)
            all_params["sigma_x2"].append(params["sigma_x2"].value)
            all_params["sigma_y2"].append(params["sigma_y2"].value)
            all_params["x"].append(star.x)
            all_params["y"].append(star.y)
        
        print("\n\n" + "SUMMARY")
        print("Number of stars successfully analyzed:", len(starlist) - skip_count)
        print(
            "sigma_x2", str(np.average([x for x in all_params["sigma_x2"] if x < 10]))
        )
        print(
            "sigma_y2", str(np.average([y for y in all_params["sigma_y2"] if y < 10]))
        )

        Table(all_params).write(output_path, format="csv", overwrite=True)
        print("Wrote to file: " + output_path)


def main():
    GeneralPSFDriver()
    

if __name__ == "__main__":
    main()
