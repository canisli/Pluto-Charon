import time
import glob

from GaussianModel import *
from Image import *
from res import config
from res.constants import *

def main():
    PlutoCharonSetupData = {}
    print("Analyzing pluto and charon on", config.date)

    files = glob.glob(config.data_folder + config.date + "/*.fits").sort()
    PlutoCharonSetupData["orig_image"] = Image(files[0])

    blob_center_x = constants[config.date]["blob_center_x"]
    blob_center_y = constants[config.date]["blob_center_y"]
    PlutoCharonSetupData["subimage"] = PlutoCharonSetupData["orig_image"].subimage(
        blob_center_x, blob_center_y, 19, 19
    )
    # estimate based off grabbing values from ds9
    PlutoCharonSetupData["init_background"] = constants[config.date]["background"]
    counts = constants[config.date]["pluto_charon_counts"]
    PlutoCharonSetupData["init_Ap"] = constants[config.date]["init_Ap_coeff"] * counts
    PlutoCharonSetupData["init_Ac"] = constants[config.date]["init_Ac_coeff"] * counts

    # average from GaussianModel.get_params
    PlutoCharonSetupData["sigma_x2"] = constants[config.date]["sigma_x2"]
    PlutoCharonSetupData["sigma_y2"] = constants[config.date]["sigma_y2"]
    PlutoCharonSetupData["subimage"].write_fits(config.output_folder + config.date + "_PC_subimage")
    print("Using sigma_x2=", PlutoCharonSetupData["sigma_x2"])
    print("Using sigma_y2=", PlutoCharonSetupData["sigma_y2"])
    locate_pluto_charon(PlutoCharonSetupData)

if __name__ == "__main__":
    main()