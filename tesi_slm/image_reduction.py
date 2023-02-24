import numpy as np
from tesi_slm.camera_masters import CameraMastersAnalyser
from astropy.io import fits

class MeasureCleanImages():
    
    def __init__(self, camera, fname_masters):
        self._cam = camera
        self._texp, self._fNframes, \
         self._master_dark, self._master_background = \
         CameraMastersAnalyser.load_camera_masters(fname_masters)
    
    def acquire_images(self, Nframes = 100):
        
        self._Nframes = Nframes
        self._cam.setExposureTime(self._texp)
        cube = self._cam.getFutureFrames(Nframes, 1) 
        self._cube_images = cube.toNumpyArray()
    
    def save_master(self, fname):
        hdr = fits.Header()
        #hdr['MAST'] = tag
        hdr['T_EX_MS'] = self._texp
        hdr['N_FR'] = self._Nframes
        fits.writeto(fname, self._cube_images, hdr)
    
    def _clean_images(self):
        tmp_clean = np.zeros(self._cube_images.shape)
        for idx in range(self._Nframes_bg):
            tmp_clean[:,:,idx]  = self._cube_images[:,:,idx] - self._master_background - self._master_dark
        self._master_background = np.median(tmp_clean, axis = 2)