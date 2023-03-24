import numpy as np 
from tesi_slm.camera_masters import CameraMastersAnalyzer
from tesi_slm.my_tools import clean_cube_images, get_index_from_image, cut_image_around_coord
from astropy.io import fits


class DiffractionEfficiencyMeasurer():
    
    def __init__(self, spoc, fname_masters):
        self._spoc = spoc
        self._spoc.change_circular_mask(RadiusInPixel=1920*0.5)
        self._texp, self._fNframes, \
         self._master_dark, self._master_background = \
         CameraMastersAnalyzer.load_camera_masters(fname_masters)
    
    def measure_diffraction_efficiency(self, c_span, j = 2, N_iter = 10):
        Nframes = 30
        self._c_span = c_span 
        self._spoc._cam.setExposureTime(self._texp)
        
        # self._spoc.set_slm_flat()
        # flat_cube = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
        # clean_flat = clean_cube_images(flat_cube, self._master_dark, self._master_background) 
        # yf, xf = get_index_from_image(clean_flat, value = clean_flat.max())
        # cut_clean_flat = cut_image_around_coord(clean_flat, yf, xf, halfside=25)
        # I_total = cut_clean_flat.sum()
        # print('I_tot:%g'%I_total)
        #

        N_amp = len(c_span)
        I_mod_values = np.zeros((N_amp, N_iter))
        I_ghost_values = np.zeros((N_amp, N_iter))
        self._mean_mod_ratio = np.zeros(N_amp)
        self._err_mod_ratio = np.zeros(N_amp)
        self._mean_ghost_ratio = np.zeros(N_amp)
        self._err_ghost_ratio = np.zeros(N_amp)
        
        coeff2apply = np.zeros(2)
        tilt_idx = j - 2
        
        for c_idx in range(N_amp):
            
            coeff2apply[tilt_idx] = c_span[c_idx]
            print('c = %g m'% c_span[c_idx])
            for i in range(N_iter):
                print('%d iter'%i)
                
                self._spoc.set_slm_flat()
                
                flat_cube = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
                clean_flat = clean_cube_images(flat_cube, self._master_dark, self._master_background) 
                if i == 0:
                    yf, xf = get_index_from_image(clean_flat, value = clean_flat.max())
                    
                cut_clean_flat = cut_image_around_coord(clean_flat, yf, xf, halfside=25)
                I_total = cut_clean_flat.sum()
                print('I_tot:%g'%I_total)
                
                self._spoc._write_zernike_on_slm(coeff2apply)
                
                tilt_cube = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
                clean_tilt = clean_cube_images(tilt_cube, self._master_dark, self._master_background)
                if i == 0:                    
                    ym, xm = get_index_from_image(clean_tilt, value = clean_tilt.max())
                    
                cut_clean_tilt = cut_image_around_coord(clean_tilt, ym, xm, halfside=25)
                cut_clean_ghost = cut_image_around_coord(clean_tilt, yf, xf, halfside=25)
                
                I_mod_values[c_idx, i] = (cut_clean_tilt.sum())/I_total
                I_ghost_values[c_idx, i] = (cut_clean_ghost.sum())/I_total
                print('mod_ratiot:%g'%I_mod_values[c_idx, i])
                print('ghost_ratio:%g'%I_ghost_values[c_idx, i])
                
            self._mean_mod_ratio[c_idx] = I_mod_values[c_idx,:].mean()
            self._err_mod_ratio[c_idx] = I_mod_values[c_idx,:].std()
            self._mean_ghost_ratio[c_idx] = I_ghost_values[c_idx,:].mean()
            self._err_ghost_ratio[c_idx] = I_ghost_values[c_idx,:].std()
        
    def show_ratios(self):
        import matplotlib.pyplot as plt
        plt.subplots(2,1,sharex=True)
        plt.subplot(2,1,1)
        plt.plot(self._c_span, self._mean_ghost_ratio, 'bo-', label = 'ghost ratio')
        plt.errorbar(self._c_span, self._mean_ghost_ratio, self._err_ghost_ratio, fmt='.b')
        plt.xlabel('c [m]')
        plt.ylabel('$<I_{ghost}> / <I_{flat}>$')
        plt.legend(loc='best')
        plt.grid(ls ='--', alpha = 0.5)
        plt.subplot(2,1,2)
        plt.plot(self._c_span, self._mean_mod_ratio, 'ro-', label = 'mod ratio')
        plt.errorbar(self._c_span, self._mean_mod_ratio, self._err_mod_ratio, fmt='.r')
        plt.xlabel('c [m]')
        plt.ylabel('$<I_{modulated}> / <I_{flat}>$')
        plt.legend(loc='best')
        plt.grid(ls ='--', alpha = 0.5)
          