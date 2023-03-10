
import numpy as np 
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
import pysilico
from tesi_slm import display_center
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D
#from astropy.io import fits


def create_devices():
    camera = pysilico.camera('localhost', 7100)
    pippo = display_center.Main230123()
    slm = pippo._mirror
    return camera, slm

class SharpPsfOnCamera():
    
    def __init__(self, camera, slm):
        self._cam = camera
        self._mirror = slm
        self._height = self._mirror.getHeightInPixels()
        self._width = self._mirror.getWidthInPixels()
        self._mirror_shape = (self._height, self._width)
        self._build_defalut_circular_mask()
        
    def _build_defalut_circular_mask(self):
        radius_in_pixel = 555
        centerYX = (self._height // 2, self._width // 2)
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = radius_in_pixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
    
    def change_circular_mask(self,
                            centerYX = (576, 960),
                            RadiusInPixel = 555):
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = RadiusInPixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
        
    def reset_default_circular_mask(self):
        self._build_defalut_circular_mask()
        
    def _write_zernike_on_slm(self,
                             zernike_coefficients_in_meters, add_wfc = True):
        zernike_builder = ZernikeGenerator(self._cmask_obj)
        image_to_display = np.zeros(self._mirror_shape)
        for j, aj in enumerate(zernike_coefficients_in_meters):
            Zj = zernike_builder.getZernike(j + 2)
            image_to_display += aj * Zj
        self._mirror.setZonalCommand(
            zonalCommand = image_to_display,
            add_correction = add_wfc)
        
    def sharp(self, j_index_to_explore, c_span , texp_ms=0.125, init_coeff = None):
        explore_jnoll = np.array(j_index_to_explore)
        #N_of_jnoll = len(explore_jnoll)
        if init_coeff is None:
            init_coeff = np.zeros(10)
        
        Namp = len(c_span)
           
        coeff2apply = init_coeff
        
        self.set_slm_flat()
        self._cam.setExposureTime(texp_ms)
        Nframes = 30
        
        for j in explore_jnoll:
            peaks = np.zeros(Namp)
            for idx_c, cj in enumerate(c_span):
                #starting from z2 up to z11
                k = int(j - 2)
                coeff2apply[k] = cj
                self._write_zernike_on_slm(coeff2apply)
                image = self.get_mean_image(Nframes)
                peaks[idx_c] = self._get_max_image(image)
            max_idx = self._get_max_index(peaks)
            coeff2apply[k] = c_span[max_idx]
        
        best_coeff = coeff2apply    
        return best_coeff
    
    def sharp_sensorlessAO(self, j_index_to_explore, c_span , texp_ms=0.125, init_coeff = None):
        explore_jnoll = np.array(j_index_to_explore)
        #N_of_jnoll = len(explore_jnoll)
        if init_coeff is None:
            init_coeff = np.zeros(10)
        
        Namp = len(c_span)
           
        coeff2apply = init_coeff
        
        self.set_slm_flat()
        self._cam.setExposureTime(texp_ms)
        Nframes = 30
        
        for j in explore_jnoll:
            peaks = np.zeros(Namp)
            for idx_c, cj in enumerate(c_span):
                #starting from z2 up to z11
                k = int(j - 2)
                coeff2apply[k] = cj
                self._write_zernike_on_slm(coeff2apply)
                image = self.get_mean_image(Nframes)
                peaks[idx_c] = image.std()
            max_idx = self._get_max_index(peaks)
            coeff2apply[k] = c_span[max_idx]
        
        best_coeff = coeff2apply    
        return best_coeff
    
    def _get_max_index(self, array1D):
        max_val = array1D.max()
        idx = np.where(array1D == max_val)[0][0]
        return idx
    
    # def _get_image_peak_and_coords(self, image):
    #     peak = image.max()
    #     ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
    #     return peak, ymax, xmax
    
    def _gaussian_fit(self, image, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
        dimy, dimx = image.shape
        y, x = np.mgrid[:dimy, :dimx]
        fitter = LevMarLSQFitter(calc_uncertainties=True)
        model = Gaussian2D(amplitude=amplitude,
                           x_mean=x_mean, y_mean=y_mean,
                           x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                           y_stddev=fwhm_y * gaussian_fwhm_to_sigma)
        fit = fitter(model, x, y, image)
        return fit
    
    def _weighted_gaussian_fit(self, image, err_ima, x_mean, y_mean, fwhm_x , fwhm_y, amplitude):
        dimy, dimx = image.shape
        y, x = np.mgrid[:dimy, :dimx]
        fitter = LevMarLSQFitter(calc_uncertainties=True)
        model = Gaussian2D(amplitude=amplitude,
                           x_mean=x_mean, y_mean=y_mean,
                           x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                           y_stddev=fwhm_y * gaussian_fwhm_to_sigma)
        w = 1/err_ima
        fit = fitter(model, x, y, z = image, weights = w)
        return fit
    
    def get_mean_image(self, Nframes2average = 30):
        ima = self._cam.getFutureFrames(1, Nframes2average)
        return ima.toNumpyArray()
    
    def _get_max_image(self, image):
        return image.max()
    
    def set_slm_flat(self, add_wfc = True):
        self._mirror.setZonalCommand(
            zonalCommand = np.zeros((self._height, self._width)),
            add_correction = add_wfc)
        
    def close_slm(self):
        self._mirror.deinitialize()
    