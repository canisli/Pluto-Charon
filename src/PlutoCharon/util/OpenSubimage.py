from res import config

import os


def main():
    os.system(
        "open "
        + config.output_folder
        + config.date
        + "/"
        + config.date
        + config.index
        + "_PC_subimage.fits"
    )


if __name__ == "__main__":
    main()
