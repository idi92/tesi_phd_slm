from plico_dm_server.controller.meadowlark_slm_1920 import \
    initialize_meadowlark_sdk, MeadowlarkSlm1920
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
import numpy as np
import pysilico
#from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

class Main230123():
    
    BLINK_DIR_ROOT = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus'
    LUT_FNAME = '\LUT Files\slm6208_635_PCIe.LUT'
    WFC_FNAME = '\WFC Files\slm6208_at635_WFC.bmp'
    WL_CALIB = 635e-9
    HOSTNAME='localhost'
    PORT = 7100
    DEFAULT_RADIUS = 570
    DEFAULT_CENTER = (576, 960)
    MAX_INTENSITY = 2000
    INTENSITY_THRESHOLD = 0.9
    DEFAULT_EXP_TIME_MILLISEC = 0.2
    
    def __init__(self):
        slm_lib, image_lib = initialize_meadowlark_sdk(self.BLINK_DIR_ROOT)
        self._mirror = MeadowlarkSlm1920(
            slm_lib,
            image_lib,
            self.BLINK_DIR_ROOT + self.LUT_FNAME,
            self.BLINK_DIR_ROOT + self.WFC_FNAME,
            self.WL_CALIB)
        self._height = self._mirror.getHeightInPixels()
        self._width = self._mirror.getWidthInPixels()
        self._mirror_shape = (self._height, self._width)
        self._cmask_obj = None
    
    def build_circular_mask_object(self, centerYX = (576, 960), RadiusInPixel = 540):
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = RadiusInPixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
        return cmask
    
    def write_zernike_on_slm(self,
                             CircularMaskObj,
                             zernike_coefficients_in_meters, add_wfc = True):
        zernike_builder = ZernikeGenerator(CircularMaskObj)
        image_to_display = np.zeros(self._mirror_shape)
        for j, aj in enumerate(zernike_coefficients_in_meters):
            Zj = zernike_builder.getZernike(j+2)
            image_to_display += aj * Zj
        self._mirror.setZonalCommand(
            zonalCommand = image_to_display,
            add_correction = add_wfc)
    
    def get_image_on_slm(self):
        return self._mirror.getZonalCommand()
        
    def _get_slm_display_center(self):
        
        xc = self._width // 2
        yc = self._height // 2
        return xc, yc
    
    def see_central_actuators(self, pixelside = 20, amp = 635e-9):
        cmd_vector = np.zeros((self._height, self._width))
        xc, yc = self._get_slm_display_center()
        cmd_vector[yc-pixelside//2:yc+pixelside//2, 
                   xc-pixelside//2:xc+pixelside//2] = amp
        self._mirror.setZonalCommand(zonalCommand = cmd_vector,
                                    add_correction = False)
    def set_slm_flat(self, add_wfc = True):
        self._mirror.setZonalCommand(
            zonalCommand = np.zeros((self._height, self._width)),
            add_correction = add_wfc)
        
    def close_slm(self):
        self._mirror.deinitialize()
    
    def get_default_circular_mask(self):
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = self.DEFAULT_RADIUS,
            maskCenter = self.DEFAULT_CENTER)
        return cmask
    
    def display_psf_comparison_wrt_slm_flat(self, cmask, j, aj, expTimeInMillisec = 0.2, add_wfc = True):
        cam = pysilico.camera(self.HOSTNAME, self.PORT)
        cam.setExposureTime(exposureTimeInMilliSeconds = expTimeInMillisec)
        self.set_slm_flat(add_wfc)
        ima = cam.getFutureFrames(1, 10)
        flat_image = ima.toNumpyArray()
        self.write_zernike_on_slm(
            CircularMaskObj = cmask,
            j = j,
            aj = aj,
            add_wfc = add_wfc)
        ima = cam.getFutureFrames(1, 10)
        zernike_image = ima.toNumpyArray()
        
        import matplotlib.pyplot as plt
        plt.subplots(1, 2, sharex=True, sharey=True)
        plt.subplot(1, 2, 1)
        plt.title('Zernike : j = %d, aj = %g m ' %(j, aj))
        plt.imshow(zernike_image, cmap = 'jet')
        plt.subplot(1, 2, 2)
        plt.title('flat : wfc = %s' %add_wfc)
        plt.imshow(flat_image, cmap = 'jet')
    
    def compute_main_abs_coeff_for_best_psf(self):
        pass
 
        