# Pluto-Charon
This project will attempt to measure the orbital period of the Pluto/Charon system by measuring the relative positions of Pluto and Charon on a series of images of the system spread over period of several months.

<img src="https://user-images.githubusercontent.com/73449574/154826770-9b3ed249-0936-4aa6-bcd3-64b136617b74.png" width="400" />

### File structure
* `data/(date in m-dd-yyyy)/`: the data for each observing night including fits files and text files containing locations of stars
* `data/pluto_locations.txt`: the coordinates of Pluto and Charon (manually found)
* `src/PlutoCharon/Image.py`: represents a fits file
* `src/PlutoCharon/IStar.py`: represents a star inside Image
* `src/PlutoCharon/GaussianModel.py`: contains Gaussian models for running least squares minimization to determine point spread function of stars and Pluto/Charon
* `src/PlutoCharon/RunStarGaussian.py`: runs Gaussian model on each star in Image
* `src/PlutoCharon/RunPlutoCharonGaussian.py`: runs Gaussian model on Pluto/Charon blob
* `src/PlutoCharon/res/`: config and constants
* `src/PlutoCharon/util/`: utility files mainly for testing and displaying results
