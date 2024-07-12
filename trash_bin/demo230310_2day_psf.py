import numpy as np 
from astropy.io import fits

class MeasurePSF():
    
    def __init__(self, spoc):
        self._spoc = spoc
            
    def measure_psf(self, Nframes, texp = 0.05, threshold_bg = 50):
        self._texp = texp
        self._Nframes = Nframes
        self._spoc._cam.setExposureTime(self._texp)
        cam_shape = self._spoc._cam.shape()
        cube = np.zeros((cam_shape[0], cam_shape[1], Nframes))
        for n in range(Nframes):
            ima = self._spoc._cam.getFutureFrames(1).toNumpyArray()
            bg = ima[ima<threshold_bg].mean()
            cube[:, :, n] = ima - bg 
         
        self._clean_images = cube
        
        
    def save_image(self, fname, z_coeff = None):
        hdr = fits.Header()
        if z_coeff is None:
            z_coeff = np.zeros(3)
        z_coeff = np.array(z_coeff)
        hdr['T_EX_MS'] = self._texp
        hdr['N_FR'] = self._Nframes
        fits.writeto(fname, self._clean_images, hdr)
        fits.append(fname, z_coeff)
    
    @staticmethod    
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        clean_images = hduList[0].data 
        z_coeff = hduList[1].data 
        Nframes = header['N_FR']
        texp = header['T_EX_MS']
        return texp, Nframes, clean_images, z_coeff
    