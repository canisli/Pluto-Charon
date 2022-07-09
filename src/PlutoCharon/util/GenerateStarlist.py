from Image import Image
from res import config


def main():
    image_path = f'{config.data_folder}/{config.date}/pluto{config.index}.fits'
    image = Image(image_path)
    
    output = f'{config.data_folder}/{config.date}/starlist{config.index}.ascii'
    image.save_starlist(output)
    print(f'Wrote starlist to {output}')


if __name__ == '__main__':
    main()
