import glob

from GaussianModel import *
from Image import *
from res import config
from res.constants import *

def main():
    print(config.data_folder + config.date + "/*.fits")
    files = glob.glob(config.data_folder + config.date + "/*.fits")
    files.sort()
    path = files[0]

    image = Image(path)
    hdul = fits.open(path)

    fwhm_arc = 3.5  # full width half maximum in arcseconds
    fwhm = (
        fwhm_arc / hdul[get_image_hdu_number(hdul)].header["CDELT1"]
    )  # fwhm in pixels
    
    PSFSetupData = {}
    PSFSetupData["star_x"] = constants[config.date]["blob_center_x"]
    PSFSetupData["star_y"] = constants[config.date]["blob_center_y"]
    PSFSetupData["orig_image"] = image
    PSFSetupData["subimage"] = PSFSetupData["orig_image"].subimage(
        constants[config.date]["blob_center_x"],
        constants[config.date]["blob_center_y"],
        19,
        19,
    )
    PSFSetupData["fwhm"] = fwhm
    PSFSetupData["avg_pixel_val"] = image.get_average_pixel_value()

    gm = StarGaussian(PSFSetupData)

    params = gm.get_params()

    print(params["a"].value)
    print(params["b"].value)
    print(params["sigma_x2"].value)
    print(params["sigma_y2"].value)


if __name__ == "__main__":
    main()