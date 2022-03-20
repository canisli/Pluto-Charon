from GaussianModel import *
import config
import constants as c

# Run a PlutoCharon gaussian on a nearby star to check for convergence
# Hopefully it will show that the two objects are superimposed

setupData = {}
print("Analyzing nearby star on ", config.date)

setupData['orig_image'] = Image("./data/" + config.date + "/pluto_V.fits")
blob_center_x = c.constants[config.date]['nearby_star_x'] # displace from given coords to center blob
blob_center_y = c.constants[config.date]['nearby_star_y']
setupData['subimage'] = setupData['orig_image'].subimage(blob_center_x, blob_center_y, 19, 19)
setupData['subimage'].write_fits('teststar.fits')
setupData['init_background'] = c.constants[config.date]['background']
counts = c.constants[config.date]['nearby_star_counts']
setupData['init_Ap'] = c.constants[config.date]['init_Ap_coeff'] * counts
setupData['init_Ac'] = c.constants[config.date]['init_Ac_coeff'] * counts

setupData['sigma_x2'] = c.constants[config.date]['sigma_x2']
setupData['sigma_y2'] = c.constants[config.date]['sigma_y2']

print("Using sigma_x2=",  setupData["sigma_x2"])
print("Using sigma_y2=",  setupData["sigma_y2"])
print("Value of get_pixel(5,5)", setupData["subimage"].get_pixel(5, 5))
locate_pluto_charon(setupData)
