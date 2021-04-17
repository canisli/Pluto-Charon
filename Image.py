from astropy.io import fits
from astropy.table import Table
import numpy as np

class Image:
	def __init__(self, file_name=None, width=None, height=None):
		if file_name is not None:
			self.file_name = file_name
			hdul = fits.open(file_name)
			self.data = hdul[1].data # assume the first extension is an image
			hdul.close()

			self.height = len(self.data)
			self.width = len(self.data[0])
		else:
			self.width = width
			self.height = height

			self.file_name = "test.fits"
			hdu1 = fits.PrimaryHDU()
			self.data = np.zeros((width,height))
			hdu2 = fits.ImageHDU(self.data)
			new_hdul = fits.HDUList([hdu1, hdu2])
			new_hdul.writeto(self.file_name, overwrite=True)
				
	def WriteFITS(self, file_name_string):
		pass	
	
	def __str__(self):
		return "Image " + str(self.file_name) + ": " 
			+ str(self.width) + " x " + str(self.height)

def main():
	image = Image(file_name="u-aur_V.fits")
	image2 = Image(width=800, height=500)
	print(image)
	print(image2)

if __name__=="__main__":
    main()
		
