import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
import pysilico
from tesi_slm import display_center
from astropy.io import fits


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
    DEFAULT_BG_DIR = 'C:/Users/labot/Desktop/misure_tesi_slm/230216'
    DEFAULT_BG_FNAME = '/230216backgroung_camera.fits'
    
    def __init__(self, camera, slm):
        self._cam = camera
        self._mirror = slm
        self._height = self._mirror.getHeightInPixels()
        self._width = self._mirror.getWidthInPixels()
        self._mirror_shape = (self._height, self._width)
        self._build_defalut_circular_mask()
        self._back_ground = None
    
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
                        image = self.get_mean_image_from_camera(
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
            
    
    def get_mean_image_from_camera(self,
                              frame_to_avarage = 30):
        # TODO : check pixel  saturation and change exptime
        #self._cam.setExposureTime(exp_time_in_millisec)
        image = self._cam.getFutureFrames(1, frame_to_avarage)
        array_image = image.toNumpyArray() 
        #saturation_reached = self._check_pixel_saturation_on_camera(array_image)
        return array_image
    
    def load_camera_background(self, fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        self._back_ground = hduList[0].data
        sigma_back_ground = hduList[1].data
        
        num_of_frames = header['N_AV_FR']
        self._texp_bg = header['T_EX_MS']
        
    
    def get_frames_from_camera(self, NumOfFrames = 100):
        image = self._cam.getFutureFrames(NumOfFrames)
        array_image = image.toNumpyArray() 
        return array_image
    
    def get_mean_and_std_from_frames(self, image):
        mean_ima = image.mean(axis = 2)
        sigma_ima = image.std(axis = 2)
        return mean_ima, sigma_ima
    
    def _subtract_background_from_image(self, mean_image):
        tmp = mean_image - self._back_ground
        sub_ima = tmp - np.median(tmp) 
        return sub_ima
    
    def _get_mean_peak_and_error_from_image(self, mean_ima, sigma_ima):
        imax = mean_ima.max()
        ymax, xmax = np.where(mean_ima==imax)[0][0], np.where(mean_ima==imax)[1][0]
        cut_image = mean_ima[ymax-1:ymax+2, xmax-1:xmax+2]
        # TO DO: try to oversample and interpolate?
        idx_list_y = np.where(np.abs(cut_image/cut_image.max())>=0.80)[0]
        idx_list_x = np.where(np.abs(cut_image/cut_image.max())>=0.80)[1]
        
        Npix = len(idx_list_x)
        
        mean_peak = cut_image[idx_list_y, idx_list_x].mean()
        cut_err_image = sigma_ima[ymax-1:ymax+2, xmax-1:xmax+2]
        sig2=cut_err_image[idx_list_y, idx_list_x]**2
        err_peak = np.sqrt(sig2.sum()/(Npix*Npix))
        return mean_peak, err_peak
     
    # This automatic function has many issues :(
    def compute_zernike_coeff2optimize_psf(self,
                                        list_of_starting_coeffs_in_meters,
                                        max_amp = 200e-9,
                                        min_amp = -200e-9):
        
        initial_coeffs = np.array(list_of_starting_coeffs_in_meters)
        num_of_coeffs = len(initial_coeffs)
        best_coeffs = initial_coeffs
        
        for idx in range(num_of_coeffs):
            j = idx + 2
            print('Turn j = %d'%j)
            best_coeffs[idx] = self._estimate_coeff(idx, best_coeffs, min_amp, max_amp)
            print('\tBest coeff found = %g'%best_coeffs[idx])
        return best_coeffs
    
    def search_zernike_coeff2optimize_psf(self, j_idx, amp_span_in_meters, init_coeff_in_meters):
        if self._back_ground is None:
            bg_fname = self.DEFAULT_BG_DIR + self.DEFAULT_BG_FNAME
            self.load_camera_background(bg_fname)
        Npoints = len(amp_span_in_meters)
        peaks = np.zeros(Npoints)
        err_peaks = np.zeros(Npoints)
        coeffs = init_coeff_in_meters
        self.set_slm_flat()
        j_index = j_idx - 2
        t_exp = self._texp_bg #0.125ms
        Nframe2average = 100
    
        for idx, amp in enumerate(amp_span_in_meters):
            coeffs[j_index] = amp
            self._write_zernike_on_slm(
                zernike_coefficients_in_meters = coeffs,
                add_wfc = True)
            self._cam.setExposureTime(0.125)
            image = self.get_frames_from_camera(Nframe2average)
            image_mean, image_sigma = self.get_mean_and_std_from_frames(image)
            ima_sub_bg = self._subtract_background_from_image(image_mean)
            
            peaks[idx], err_peaks[idx] = \
            self._get_mean_peak_and_error_from_image(ima_sub_bg, image_sigma)
            #peaks[idx] = self._get_intensity_peak(image)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(amp_span_in_meters, peaks, 'ko', label = 'texp= %g ms'%t_exp)
        plt.errorbar(amp_span_in_meters, peaks , err_peaks, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d} [m]$'%j_idx)
        plt.ylabel('Peak Intensity')
        plt.grid()
        plt.legend(loc='best') 
        
    
    
    def show_psf_comparison_wrt_slm_flat(self,
                                         z_coeff_list_in_meters,
                                         texp_in_ms = 1,
                                         Nframe2average = 100,
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
        zernike_image = self._subtract_background_from_image(zernike_image)
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