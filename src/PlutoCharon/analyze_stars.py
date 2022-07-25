import logging
import sys

from astropy.io import fits
from astropy.table import Table
import numpy as np

from gaussian_model import StarGaussian
from image import Image, get_image_hdu_number, distance
from istar import IStar
from res import config

log = logging.getLogger('stars')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    path = f'{config.data_folder}/{config.date}/pluto{config.index}.fits'
    starlist_path = f'{config.data_folder}/{config.date}/starlist{config.index}.ascii'

    starlist = Table.read(starlist_path, format='csv')
    output_path = (
        f'{config.output_folder}/{config.date}/gaussian_results{config.index}.ascii'
    )

    image = Image(path)
    hdul = fits.open(path)

    # full width half maximum in arcseconds (based on location in Rhode Island)
    fwhm_arc = 3.5
    fwhm = fwhm_arc / hdul[get_image_hdu_number(hdul)].header['CDELT1']

    all_params = {
        'star': [],
        'a': [],
        'bg': [],
        'sigma_x2': [],
        'sigma_y2': [],
        'x': [],
        'y': [],
    }

    skip_count = 0

    for i in range(len(starlist)):
        star = IStar(table_row=starlist[i])
        # filter out bad stars
        skip = False
        if not isinstance(star.counts, str):  # if counts not 'N/A'
            if star.counts < 0:
                skip = True
        for j in range(len(starlist)):
            # ignore stars that are within certain amount of pixels of the current star 
            # to avoid interference
            star2 = IStar(table_row=starlist[j])
            if (
                i != j
                and distance(star.x, star2.x, star.y, star2.y) < config.min_distance
            ):
                skip = True
        # stars that are too close to the border
        if star.x < 30 or star.x > image.width - 30:
            skip = True
        if star.y < 30 or star.y > image.height - 30:
            skip = True
        if skip:
            skip_count += 1
            continue

        log.info(f'<{i + 1}> \n {starlist[i]}')
        psf_setup_data = {}
        psf_setup_data['star_x'] = star.x
        psf_setup_data['star_y'] = star.y
        psf_setup_data['image'] = image
        psf_setup_data['subimage'] = psf_setup_data['orig_image'].subimage(
            star.x + 1,
            star.y + 1,
            19,
            19,
        )
        psf_setup_data['fwhm'] = fwhm
        psf_setup_data['bg'] = image.get_average_pixel_value()
        psf_setup_data['a'] = np.max(image.data) - psf_setup_data['bg']
        psf_setup_data['sigma'] = psf_setup_data['fwhm'] / 2.355

        gm = SingleGaussian(psf_setup_data)

        params = gm.run_minimizer()
        all_params['star'].append(star.star_name)
        all_params['a'].append(params['a'].value)
        all_params['bg'].append(params['bg'].value)
        all_params['sigma_x2'].append(params['sigma_x2'].value)
        all_params['sigma_y2'].append(params['sigma_y2'].value)
        all_params['x'].append(star.x)
        all_params['y'].append(star.y)

    log.info('\nSUMMARY')
    log.info(f'Number of stars successfully analyzed: {len(starlist)- skip_count}')
    log.info(
        'Average sigma_x2',
        str(np.average([x for x in all_params['sigma_x2'] if x < 10])),
    )
    log.info(
        'Average sigma_y2',
        str(np.average([y for y in all_params['sigma_y2'] if y < 10])),
    )

    Table(all_params).write(
        output_path, format='ascii.fixed_width_two_line', overwrite=True
    )
    log.info(f'Wrote to file: {output_path}')


if __name__ == '__main__':
    main()
