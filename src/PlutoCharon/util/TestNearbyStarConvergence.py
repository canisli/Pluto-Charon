############################
# outdated, need to update #
############################

from Image import *
from GaussianModel import *
from res import config
from res.constants import *


# Run a PlutoCharon gaussian on a nearby star to check for convergence
# Hopefully it will show that the two objects are superimposed

setupData = {}
print("Analyzing nearby star on", config.date)

star_number = 3  # pick from multiple nearby stars in constants file
star_name = "nearby_star" + str(star_number)

setupData["orig_image"] = Image(config.data_folder + config.date + "/pluto_V.fits")
blob_center_x = constants[config.date][
    star_name + "_x"
]  # displace from given coords to center blob
blob_center_y = constants[config.date][star_name + "_y"]
setupData["subimage"] = setupData["orig_image"].subimage(
    blob_center_x, blob_center_y, 19, 19
)
setupData["subimage"].write_fits("teststar.fits")
setupData["init_background"] = constants[config.date]["background"]
counts = constants[config.date][star_name + "_counts"]
setupData["init_Ap"] = constants[config.date]["init_Ap_coeff"] * counts
setupData["init_Ac"] = constants[config.date]["init_Ac_coeff"] * counts

setupData["sigma_x2"] = constants[config.date]["sigma_x2"]
setupData["sigma_y2"] = constants[config.date]["sigma_y2"]

print("Using sigma_x2=", setupData["sigma_x2"])
print("Using sigma_y2=", setupData["sigma_y2"])
locate_pluto_charon(setupData)
