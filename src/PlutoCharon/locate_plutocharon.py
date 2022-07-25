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
logging.getLogger('parso.python.diff').disabled = True # turn off IPython debugging

def locate_pluto_charon(psfinit, subimage, verbose=False):
    """
    Run PlutoCharonGaussian over multiple initial conditions and
    return dict of results
    """

    psfinit['x_p'] = subimage.width / 2
    psfinit['y_p'] = subimage.height / 2
    psfinit['dx'] = 2.4836
    psfinit['dy'] = -0.467
    pluto_charon = DoubleGaussian(psfinit, subimage)
    pluto_charon.set_vary(['x_p', 'y_p', 'dx', 'dy', 'a_p', 'bg', 'theta'])
    pluto_charon.set_limits('a_p', 100.0, 5000.0)
    pluto_charon.set_limits('bg', 400.0, 600)
    pluto_charon.set_limits('x_p', 10, subimage.width-10)
    pluto_charon.set_limits('y_p', 10, subimage.height-10)
    pluto_charon.set_limits('dx', -8.5, 8.5)
    pluto_charon.set_limits('dy', -8.5, 8.5)
    pluto_charon.set_limits('theta', 0, 2 * np.pi)

    params = pluto_charon.run_minimizer()
    log.info(pluto_charon.get_result())
    if verbose:
        log.info(fit_report(params, show_correl=False))

    return params


def main():
    locations = Table(
        {
            'index': [],
            'x_p': [],
            'y_p': [],
            'dx': [],
            'dy': [],
            'bg': [],
            'A_p': [],
            'A_c': [],
            'sigma_x': [],
            'sigma_y': [],
            'theta (deg)': [],
        },
        dtype=(
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ),
    )
    output_path = f'{config.output_folder}/{config.date}/locations.csv'
    for index in config.indices:
        psfinit = {}
        entry = config.date + index
        log.info(f'Analyzing pluto and charon on {entry}')

        path = f'{config.data_folder}/{config.date}/pluto{index}.fits'

        blob_center_x = constants[entry]['blob_center_x']
        blob_center_y = constants[entry]['blob_center_y']
        subimage = Image(path).subimage(blob_center_x, blob_center_y, 60, 60)
        psfinit['bg'] = constants[entry]['background']
        psfinit['a_p'] = constants[entry]['init_Ap_coeff'] * (
            subimage.max_pixel() - constants[entry]['background']
        )
        psfinit['sigma_x'] = constants[entry]['sigma_x']
        psfinit['sigma_y'] = constants[entry]['sigma_y']
        psfinit['theta'] = 19.875 * np.pi / 180

        psfinit = locate_pluto_charon(psfinit, subimage, verbose=True)
        locations.add_row(
            [
                index[-1],
                psfinit['x_p'].value,
                psfinit['y_p'].value,
                psfinit['dx'].value,
                psfinit['dy'].value,
                psfinit['bg'].value,
                psfinit['a_p'].value,
                psfinit['a_p'].value * 0.18501,
                psfinit['sigma_x'].value,
                psfinit['sigma_y'].value,
                (psfinit['theta'].value * 180 / np.pi) % 360,
            ]
        )
        subimage.write_fits(f'{config.output_folder}/{config.date}/{entry}_PC_subimage')
    log.info(f'Wrote results to {output_path}')
    Table(locations).write(output_path, format='csv', overwrite=True)


if __name__ == '__main__':
    main()
