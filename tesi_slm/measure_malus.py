import numpy as np 
from tesi_slm.camera_masters import CameraMastersAnalyzer
from astropy.io import fits

class MeasureIntensity():
    
    def __init__(self, spoc, fname_masters):
        self._spoc = spoc
        
        self._texp, self._fNframes, \
         self._master_dark, self._master_background = \
         CameraMastersAnalyzer.load_camera_masters(fname_masters)
    
    def iterate_measure_for_angle(self, angle, N_iter=100):
        self._angle = angle
        Nframes = 100
        self._spoc._cam.setExposureTime(self._texp)
        self._spoc.set_slm_flat()
        for t in range(N_iter):
            print('iter %d'%t)
            flat_ima = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
            clean_flat, err = self._get_clean_mean_image(flat_ima)
            #peak, ymax, xmax = self._get_image_peak_and_coords(clean_flat)
            roi = self._cut_image_around_coord(clean_flat, 574, 461)
            err_roi = self._cut_image_around_coord(err, 574, 461)
            self._I_roi_val = roi.sum()
            self._I_err = np.sqrt((err_roi**2).sum())
              
    def _get_clean_mean_and_std_image(self, cube_ima):
        Nframes  = cube_ima.shape[-1]
        for n in range(Nframes):
                cube_ima[:, :, n] = cube_ima[:,:,n] - self._master_background \
                 - self._master_dark
        return cube_ima.mean(axis=-1),cube_ima.std(axis=-1)
    
    def _cut_image_around_coord(self, image, yc, xc):
        cut_image = image[yc-25:yc+25, xc-25:xc+25]
        return cut_image
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def save(self, fname):
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['ANG'] = self._angle
        data = np.array([self._I_roi_valroi, self._I_roi_err])
        fits.writeto(fname, data, hdr)