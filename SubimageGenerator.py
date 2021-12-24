# Generate subimages from an image file
# Useful for debugging GaussianModel
# Assumes that a starlist.csv file has already been created in the same directory as the fits file

import glob
import tempfile
import subprocess
from pathlib import Path

from astropy.table import Table

from Image import Image
from IStar import IStar
import config

SL = 19  # side length of subimage
# home = str(Path.home())
# path = home + "/dev/Pluto-Charon/data/"
path = config.data_folder

outside = True
inside = True

while outside:
    i = 0
    fits = sorted(
        [x for x in glob.glob(path + "**/*.fits", recursive=True) if "1x1" not in x]
    )
    print("\nFits files:")
    for f in fits:
        print(f + " [" + str(i) + "]")
        i += 1
    fn = int(input("\nChoose file: "))

    image = Image(fits[fn])

    while inside:
        starlist = Table.read(
            fits[fn][0 : fits[fn].rfind("/")] + "/starlist.csv", format="csv"
        )

        print("\nFirst 10 stars:")
        for i in range(10):
            star = IStar(table_row=starlist[i])
            entry = star.star_name + " (" + str(star.x) + ", " + str(star.y) + ")"
            print(entry + ((58 - len(entry)) * " ") + "[" + str(i + 1) + "]")
        sn = int(input("\nChoose star: "))

        star = IStar(table_row=starlist[sn - 1])
        name = star.star_name
        if name == "N/A":
            name = "noname"
        temp = tempfile.NamedTemporaryFile(prefix=name, suffix=".fits")
        image.subimage(star.x + 1, star.y + 1, 19, 19).write_fits(temp.name)
        subprocess.call(["open", temp.name])
