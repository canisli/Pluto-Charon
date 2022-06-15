from Image import Image
from res import config


def main():
    path = config.data_folder + config.date + "/pluto" + config.index + ".fits"
    image = Image(path)
    image.save_starlist(
        config.data_folder + config.date + "/starlist" + config.index + ".csv"
    )


if __name__ == "__main__":
    main()
