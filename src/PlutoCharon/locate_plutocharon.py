import sys
import logging

from lmfit import fit_report
from astropy.table import Table
import numpy as np

from gaussian_model import DoubleGaussian
from image import Image
from res import config
from res.constants import constants

log = logging.getLogger('locate')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def locate_pluto_charon(psf_setup_data, verbose=False):
    """
    Run PlutoCharonGaussian over multiple initial conditions and
    return dict of results
    """

    psf_setup_data['x_p'] = psf_setup_data['image'].width/2
    psf_setup_data['y_p'] = psf_setup_data['image'].height/2
    psf_setup_data['dx'] = -4
    psf_setup_data['dy'] = 0
    pluto_charon = DoubleGaussian(psf_setup_data)
    params = pluto_charon.get_params()
    # log.info('Locations after fitting')
    # log.info(f'Sigmas {params["sigma_x"].value}, {params["sigma_y"].value}')
    # log.info(f'Pluto {params["x_p"].value}, {params["y_p"].value}')
    # log.info(f'Charon {params["x_c"].value}, {params["y_c"].value}')
    log.info(pluto_charon.get_result())
    if verbose:
        log.info(fit_report(params, show_correl=False))
    
    return params

def main():
    locations = Table({
        'index': [],
        'x_p': [], 'y_p': [],
        'dx': [], 'dy': [],
        'bg': [],
        'A_p': [], 'A_c': [],
        'sigma_x': [], 'sigma_y': [],
        'theta (deg)': [],},
        dtype=(str, float, float, float, float, float, float, float, float, float, float)
    )
    output_path = f'{config.output_folder}/{config.date}/locations.csv'
    for index in config.indices:
        psf_setup_data = {}
        entry = config.date + index
        log.info(f'Analyzing pluto and charon on {entry}')

        path = f'{config.data_folder}/{config.date}/pluto{index}.fits'
        
        blob_center_x = constants[entry]['blob_center_x']
        blob_center_y = constants[entry]['blob_center_y']
        subimage = Image(path).subimage(
            blob_center_x, blob_center_y, 30, 30
        )
        psf_setup_data['image'] = subimage
        psf_setup_data['init_background'] = constants[entry]['background']
        psf_setup_data['init_Ap'] = constants[entry]['init_Ap_coeff'] * (subimage.max_pixel() - constants[entry]['background'])
        psf_setup_data['init_Ac'] = constants[entry]['init_Ac_coeff'] * (subimage.max_pixel() - constants[entry]['background'])
        psf_setup_data['sigma_x'] = constants[entry]['sigma_x']
        psf_setup_data['sigma_y'] = constants[entry]['sigma_y']
        psf_setup_data['theta'] = 22 * np.pi/180
        
        params = locate_pluto_charon(psf_setup_data, verbose=True)
        locations.add_row([index[-1], params['x_p'].value, params['y_p'].value,
                        params['dx'].value, params['dy'].value,
                        params['bg'].value,
                        params['a_p'].value, params['a_c'].value,
                        params['sigma_x'].value, params['sigma_y'].value,
                        (params['theta'].value * 180/np.pi) % 360])
        subimage.write_fits(
            f'{config.output_folder}/{config.date}/{entry}_PC_subimage'
        )
    log.info(f'Wrote results to {output_path}')
    Table(locations).write(output_path, format='csv', overwrite=True)

if __name__ == '__main__':
    main()
