import numpy as np 
from astropy.io import fits
from tesi_slm import psf_on_camera_optimizer

class TiltedPsfMeasurer():
    
    FRAMES_PER_TILT = 100
    
    def __init__(self):
        cam, mirror = psf_on_camera_optimizer.create_devices()
        self._poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
        self._cam = cam
    
    def measure_tilted_psf(self, j, c_span, texp = 0.125, init_coeff = None):
        self._texp = texp
        self._j_noll_idx = j
        self._c_span = c_span
        j_index = j - 2
        
        if init_coeff is None:
            #first 11 zernike starting from Z2
            init_coeff = np.zeros(9)
        self._init_coeff = np.array(init_coeff)
        
        Nmodes = len(c_span)
        self._poco._write_zernike_on_slm(init_coeff)
        coeff = init_coeff.copy()
        
        
        frame_shape = self._cam.shape()
        self._images_4d = np.zeros((Nmodes, frame_shape[0],frame_shape[1],self.FRAMES_PER_TILT))
        self._cam.setExposureTime(texp)
        
        for idx, amp in enumerate(c_span):
            coeff[j_index] = amp
            self._poco._write_zernike_on_slm(coeff)
            self._images_4d[idx] = self._poco.get_frames_from_camera(
                NumOfFrames = self.FRAMES_PER_TILT)
        
            
    def save_measures(self, fname):
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['N_AV_FR'] = self.FRAMES_PER_TILT
        hdr['Z_J'] = self._j_noll_idx
         
        fits.writeto(fname, self._images_4d, hdr)
        
        fits.append(fname, self._c_span)
        fits.append(fname,self._init_coeff)
        
    @staticmethod    
    def load_measures(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        images_4d = hduList[0].data
        c_span = hduList[1].data
        init_coeff = hduList[2].data
            
        Nframes = header['N_AV_FR']
        texp = header['T_EX_MS']
        j_noll = header['Z_J']
        return images_4d, c_span, Nframes, texp, j_noll, init_coeff
             
            