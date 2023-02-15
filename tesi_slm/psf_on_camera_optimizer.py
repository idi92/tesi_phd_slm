import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
import pysilico
from prove_su_slm import display_center


#from plico_dm import deformableMirror
def create_devices():
    camera = pysilico.camera('localhost', 7100)
    pippo = display_center.Main230123()
    slm = pippo._mirror
    return camera, slm

class PsfOnCameraOptimizer():
    DEFAULT_EXPTIME_MILLISEC = 2
    # TODO: estimate camera linearity and the choose proper parameters
    DEFAULT_MAX_INTENSITY = 3000
    INTENSITY_TRESHOLD = 0.97
    
    def __init__(self, camera, slm):
        self._cam = camera
        self._mirror = slm
        self._height = self._mirror.getHeightInPixels()
        self._width = self._mirror.getWidthInPixels()
        self._mirror_shape = (self._height, self._width)
        self._build_defalut_circular_mask()
    
    def _build_defalut_circular_mask(self):
        radius_in_pixel = self._height // 2 - 5
        centerYX = (self._height // 2, self._width // 2)
        cmask = CircularMask(
            frameShape = (self._height, self._width),
            maskRadius = radius_in_pixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
    
    def change_circular_mask(self,
                            centerYX = (576, 960),
                            RadiusInPixel = 540):
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
        
    def _estimate_coeff(self, idx, coeffs, min_amp, max_amp):
    
        if coeffs[idx] == 0:
            estimated_coeff = 0.
        else:
            texp = 2 #millisec
            min_coeff = min_amp
            max_coeff = max_amp
            Npoints = 11
            continue_opt_loop = True
            saturation_reached = True
            
            while(continue_opt_loop is True):
                amp_span = np.linspace(min_coeff, max_coeff, Npoints)
                peaks = np.zeros(Npoints)
                print('amp_span\n')
                print(amp_span)
                
                while(saturation_reached is True):
                    for k, amp in enumerate(amp_span):
                        coeffs[idx] = amp
                        self._write_zernike_on_slm(
                            zernike_coefficients_in_meters = coeffs,
                            add_wfc = True)
                        self._cam.setExposureTime(texp)
                        image = self.get_image_from_camera(
                            frame_to_avarage = 30)
                        
                        peaks[k] = self._get_intensity_peak(image)
                    print('measured peaks:\n')
                    
                    print(peaks)
                    saturation_reached = self._check_pixel_saturation_on_camera(peaks)
                    if(saturation_reached is True):
                        print('reducing texp\n')
                        texp *= 0.25
                    else:
                        saturation_reached = False
                        
                local_max_found = self._check_peaks_pattern(peaks)
                peak_max_idx = self._get_idx_relative2max_peak(peaks) 
                if(local_max_found is False):
                    print('local max not found: increasing amp_span')
                    max_coeff = amp_span[peak_max_idx] + 100e-9
                    min_coeff = amp_span[peak_max_idx] - 100e-9 
                    continue_opt_loop = True
                    saturation_reached = True
                    
                if(local_max_found is True):
                    amp_threshold =np.abs((amp_span[peak_max_idx+1]-amp_span[peak_max_idx-1])
                                          /amp_span[peak_max_idx])
                    #amp_threshold = (amp_span.max()-amp_span.min())/amp_span.mean()
                    if(amp_threshold < 0.1):
                        print('stopping opt')
                        estimated_coeff = amp_span[peak_max_idx]
                        continue_opt_loop = False
                    else:
                        print('local max found: shrinking amp_span around max')
                        max_coeff = amp_span[peak_max_idx  + 1]
                        min_coeff = amp_span[peak_max_idx  - 1]
                        continue_opt_loop = True
                        saturation_reached = True
                        
        return estimated_coeff
    
    def _get_idx_relative2max_peak(self, peaks):
        return np.where(peaks==peaks.max())[0][0]
    
    def _check_peaks_pattern(self, peaks):
        max_on_extremeA = (peaks.max()==peaks[0])
        max_on_extremeB = (peaks.max()==peaks[-1])
        decreasing_pattern = np.all(peaks[:-1]>=peaks[1:])
        increasing_pattern = np.all(peaks[:-1]<=peaks[1:])
        if(decreasing_pattern or increasing_pattern or 
           max_on_extremeA or max_on_extremeB):
            local_max_found = False
        # decreasing_pattern = np.all(peaks[:-1]>=peaks[1:])
        # increasing_pattern = np.all(peaks[:-1]<=peaks[1:])
        # if(decreasing_pattern or increasing_pattern):
        #     local_max_found = False
        else:
            local_max_found = True
        return local_max_found
   
    def _get_intensity_peak(self, image):
        # TODO: compute in a proper way the peak intensity
        return image.max()
    
    def _get_a_better_estimate_of_peak_intensity(self, image):
        imax = image.max()
        ymax, xmax = np.where(image==imax)[0][0], np.where(image==imax)[1][0]
        cut_image = image[ymax-1:ymax+2, xmax-1:xmax+2]
        # TO DO: try to oversample and interpolate?
        idx_list_y = np.where(np.abs(cut_image/cut_image.max())>=0.80)[0]
        idx_list_x = np.where(np.abs(cut_image/cut_image.max())>=0.80)[1]
        image_peak = cut_image[idx_list_y, idx_list_x].mean()
        x_mean = np.mean(xmax-1 + idx_list_x)
        y_mean = np.mean(ymax-1 + idx_list_y)
        return image_peak, y_mean, x_mean 
    
    
    def _check_pixel_saturation_on_camera(self, image):
        max_intensity =  image.max()
        if max_intensity >= self.INTENSITY_TRESHOLD *self.DEFAULT_MAX_INTENSITY:
            saturation_status = True
            # exp_time_in_millisec = self._cam.exposureTime() * 0.5
            # self.get_image_from_camera(exp_time_in_millisec)
        else:
            saturation_status = False
        return saturation_status
            
    
    def get_image_from_camera(self,
                              frame_to_avarage = 30):
        # TODO : check pixel  saturation and change exptime
        #self._cam.setExposureTime(exp_time_in_millisec)
        image = self._cam.getFutureFrames(1, frame_to_avarage)
        array_image = image.toNumpyArray() 
        #saturation_reached = self._check_pixel_saturation_on_camera(array_image)
        return array_image
    
    def compute_zernike_coeff2optimize_psf(self,
                                        list_of_starting_coeffs_in_meters,
                                        max_amp = 200e-9,
                                        min_amp = -200e-9):
        
        initial_coeffs = np.array(list_of_starting_coeffs_in_meters)
        num_of_coeffs = len(initial_coeffs)
        best_coeffs = initial_coeffs
        
        for idx in range(num_of_coeffs):
            j = idx +2
            print('Turn j = %d'%j)
            best_coeffs[idx] = self._estimate_coeff(idx, best_coeffs, min_amp, max_amp)
            print('\tBest coeff found = %g'%best_coeffs[idx])
        return best_coeffs
    
    def show_psf_comparison_wrt_slm_flat(self,
                                         z_coeff_list_in_meters,
                                         texp_in_ms = 1,
                                         Nframe2average = 30,
                                         add_wfc = True):
        self._cam.setExposureTime(exposureTimeInMilliSeconds = texp_in_ms)
        self.set_slm_flat(add_wfc = add_wfc)
        ima = self._cam.getFutureFrames(1, Nframe2average)
        flat_image = ima.toNumpyArray()
        self._write_zernike_on_slm(
            zernike_coefficients_in_meters = z_coeff_list_in_meters,
            add_wfc = add_wfc)
        ima = self._cam.getFutureFrames(1, Nframe2average)
        zernike_image = ima.toNumpyArray()
        import matplotlib.pyplot as plt
        plt.subplots(1, 2, sharex=True, sharey=True)
        plt.subplot(1, 2, 1)
        plt.title('PSF from Zernike')
        plt.imshow(zernike_image, cmap = 'jet', vmax=zernike_image.max(), vmin = zernike_image.min())
        plt.colorbar(orientation="horizontal", pad = 0.05)
        plt.subplot(1, 2, 2)
        plt.title('PSF from flat : wfc = %s' %add_wfc)
        plt.imshow(flat_image, cmap = 'jet', vmax=flat_image.max(), vmin = flat_image.min())
        plt.colorbar(orientation="horizontal", pad=0.05)
        
    def set_slm_flat(self, add_wfc = True):
        self._mirror.setZonalCommand(
            zonalCommand = np.zeros((self._height, self._width)),
            add_correction = add_wfc)
        
    def close_slm(self):
        self._mirror.deinitialize()