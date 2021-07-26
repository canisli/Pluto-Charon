from lmfit import Parameter
import Image
from Image import Image

class GaussianModel():
    def __init__(self, center_x, center_y, image, fwhm, average_pixel_value):
        self.center_x = center_x
        self.center_y = center_y
        self.image = image
        self.fwhm = fwhm
        self.average_pixel_value = average_pixel_value

    def get_sigma(self):
        pass

def main():
    image = Image("./pluto_V.fits")
    starlist_path = ""
    a = Parameter("A")
    b = Parameter("B", image.get_average_pixel_value())
    sigma_x2 = Parameter("")

if __name__ == "__main__":
    main()
