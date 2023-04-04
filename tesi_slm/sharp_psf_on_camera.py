import numpy as np 
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
import pysilico
from plico_dm import deformableMirror
#from tesi_slm import display_center
from tesi_slm.camera_masters import CameraMastersAnalyzer
from tesi_slm.my_tools import clean_cube_images,\
 cut_image_around_coord, get_index_from_array



def create_devices():
    camera = pysilico.camera('localhost', 7100)
    slm = deformableMirror('localhost', 7000)
    return camera, slm

class SharpPsfOnCamera():
    
    def __init__(self, camera, slm, fname_masters = None):
        self._cam = camera
        self._mirror = slm
        self._height = 1152 
        self._width = 1920 
        self._n_act = self.get_number_of_slm_pixel()
        self._mirror_shape = (self._height, self._width)
        self._build_defalut_circular_mask()
        self.load_masters(fname_masters)
        
    def _build_defalut_circular_mask(self):
        radius_in_pixel = 555
        centerYX = (self._height // 2, self._width // 2)
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = radius_in_pixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
    
    def change_circular_mask(self,
                            centerYX = (550, 853),
                            RadiusInPixel = 569):
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = RadiusInPixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
        
    def reset_default_circular_mask(self):
        self._build_defalut_circular_mask()
        
    def _create_opd_map_from_zernike_coeff(
            self, zernike_coefficients_in_meters):
        '''
        
        to be used with slm.set_shape to apply a given surface
        deformation expressed in Zernike coefficients
        
        Returns
        -------
        image_to_display: numpy.array
            map of wavefront corresponding to the given Zernike coefficients
        '''
        zernike_builder = ZernikeGenerator(self._cmask_obj)
        image_to_display = np.zeros(self._mirror_shape)
        image_to_display = np.ma.array(data = image_to_display, mask = self._cmask_obj.mask())
        for j, aj in enumerate(zernike_coefficients_in_meters):
            Zj = zernike_builder.getZernike(j + 2)
            image_to_display += aj * Zj
        return image_to_display
        
    def write_zernike_on_slm(self,
                             zernike_coefficients_in_meters):
        image_to_display = self._create_opd_map_from_zernike_coeff(
            zernike_coefficients_in_meters)
        
        image_vector = np.reshape(image_to_display, (self._n_act,), 'C')
        self._mirror.set_shape(image_vector)
        
    
    def select_sharpening_roi(self, yc, xc, half_side):
        self._yc_roi =  yc
        self._xc_roi = xc
        self._halfside = half_side
    
    def sharp_in_roi(self, j_index_to_explore, c_span , texp_in_ms, Nframe2average=10,  init_coeff = None, method = 'max'):
        
        explore_jnoll = np.array(j_index_to_explore)
        N_of_jnoll = len(explore_jnoll)
        
        assert explore_jnoll.max() <= 11, 'The sharpening function works only'\
        ' up to j =11, avoiding piston j=1'
        assert N_of_jnoll <= 10, 'The sharpening function works only'\
        ' up to j =11, avoiding piston j=1. Input length %d expected <=10'%N_of_jnoll
            
        if init_coeff is None:
            init_coeff = np.zeros(10)
        
        if method == 'max':
            merit_function = self._get_max_image
        if method == 'std':
            merit_function = np.std
        Namp = len(c_span)
        
        self._texp = texp_in_ms
        if(self._texp != self._texp_master):
            print('WARNING: the selected exposure time (t_exp = %g ms) is different from the '\
                  'one used to measure dark and background (t_m = %g ms)\n'\
                  'NOTE: if t_m = 0s, image reduction is not performed.'
                  %(self._texp, self._texp_master))
        coeff2apply = init_coeff.copy()
        
        self.set_slm_flat()
        self._cam.setExposureTime(self._texp)
        Nframes =  Nframe2average
        self._merit_par = np.zeros((10, Namp))
        for j in explore_jnoll:
            
            for idx_c, cj in enumerate(c_span):
                #starting from z2 up to z11
                k = int(j - 2)
                coeff2apply[k] = cj
                self.write_zernike_on_slm(coeff2apply)
                image = self._cam.getFutureFrames(Nframes, 1).toNumpyArray()
                mean_image = clean_cube_images(image, self._master_dark, self._master_background)
                image_roi = cut_image_around_coord(mean_image, self._yc_roi, self._xc_roi, self._halfside)
                self._merit_par[k, idx_c] = merit_function(image_roi)
            max_idx = get_index_from_array(self._merit_par[k], value = self._merit_par[k].max())
            coeff2apply[k] = c_span[max_idx]
        
        best_coeff = coeff2apply    
        return best_coeff
    
    def plot_merit_values(self, c_span, j_index_to_explore):
        explore_jnoll = np.array(j_index_to_explore)
        import matplotlib.pyplot as plt
        plt.figure()
        for j in explore_jnoll:
            k = int(j-2)
            plt.plot(c_span, self._merit_par[k],'o-',label='j=%d'%j)
        plt.xlabel('$c_j [m]$')
        plt.ylabel('merit value')
        plt.grid(ls='--',alpha = 0.4)
        plt.legend(loc='best')
    
    def _get_max_image(self, image):
        return image.max()
    
    def set_slm_flat(self):
        self._mirror.set_shape(np.zeros(self._n_act))
   
    
    def load_masters(self, fname_masters = None):
        
        if fname_masters is None:
            self._texp_master = 0
            self._master_dark = 0
            self._master_background = 0
        else:
            self._texp_master, fNframes, \
            self._master_dark, self._master_background = \
            CameraMastersAnalyzer.load_camera_masters(fname_masters)
    
    def get_number_of_slm_pixel(self):
        return self._mirror.get_number_of_actuators()
    
    