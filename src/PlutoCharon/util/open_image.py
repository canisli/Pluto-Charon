from res import config

import os


def main():
    os.system(f'open {config.data_folder}/{config.date}/pluto{config.index}.fits')


if __name__ == '__main__':
    main()
