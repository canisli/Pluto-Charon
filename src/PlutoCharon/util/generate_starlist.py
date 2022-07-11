from image import Image
from res import config


def main():
    for index in config.indices:
        image_path = f'{config.data_folder}/{config.date}/pluto{index}.fits'
        image = Image(image_path)
        
        output = f'{config.data_folder}/{config.date}/starlist{index}.ascii'
        image.save_starlist(output)
        print(f'Wrote starlist to {output}')


if __name__ == '__main__':
    main()
