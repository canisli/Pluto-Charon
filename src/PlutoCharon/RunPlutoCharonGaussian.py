from GaussianModel import *
from Image import *
from res import config
from res.constants import *


def main():
    PlutoCharonSetupData = {}
    entry = config.date + config.index
    print("Analyzing pluto and charon on", entry)

    path = config.data_folder + config.date + "/pluto" + config.index + ".fits"
    output_path = (
        config.output_folder + config.date + "/locations" + config.index + ".csv"
    )
    PlutoCharonSetupData["orig_image"] = Image(path)

    blob_center_x = constants[entry]["blob_center_x"]
    blob_center_y = constants[entry]["blob_center_y"]
    PlutoCharonSetupData["subimage"] = PlutoCharonSetupData["orig_image"].subimage(
        blob_center_x, blob_center_y, 19, 19
    )
    # estimate based off grabbing values from ds9
    PlutoCharonSetupData["init_background"] = constants[entry]["background"]
    counts = constants[entry]["pluto_charon_counts"]
    PlutoCharonSetupData["init_Ap"] = constants[entry]["init_Ap_coeff"] * counts
    PlutoCharonSetupData["init_Ac"] = constants[entry]["init_Ac_coeff"] * counts

    # average from GaussianModel.get_params
    PlutoCharonSetupData["sigma_x2"] = constants[entry]["sigma_x2"]
    PlutoCharonSetupData["sigma_y2"] = constants[entry]["sigma_y2"]
    PlutoCharonSetupData["subimage"].write_fits(
        config.output_folder + config.date + "/" + entry + "_PC_subimage"
    )
    print("Using sigma_x2=", PlutoCharonSetupData["sigma_x2"])
    print("Using sigma_y2=", PlutoCharonSetupData["sigma_y2"])

    locations = locate_pluto_charon(PlutoCharonSetupData)
    Table(locations).write(output_path, format="csv", overwrite=True)


if __name__ == "__main__":
    main()
