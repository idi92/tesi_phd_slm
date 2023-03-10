import numpy as np
from tesi_slm.camera_masters import CameraMastersAnalyzer
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D

class MeasureCleanImages():
    
    def __init__(self, camera, fname_masters):
        self._cam = camera
        self._texp, self._fNframes, \
         self._master_dark, self._master_background = \
         CameraMastersAnalyzer.load_camera_masters(fname_masters)
    
    def acquire_images(self, Nframes = 100):
        
        self._Nframes = Nframes
        self._cam.setExposureTime(self._texp)
        cube = self._cam.getFutureFrames(Nframes, 1) 
        self._cube_images = cube.toNumpyArray()
    
    def save_image(self, fname, z_coeff = None):
        hdr = fits.Header()
        if z_coeff is None:
            z_coeff = np.zeros(3)
        z_coeff = np.array(z_coeff)
        hdr['T_EX_MS'] = self._texp
        hdr['N_FR'] = self._Nframes
        fits.writeto(fname, self._clean_images, hdr)
        fits.append(fname, z_coeff)
        
    def clean_images(self):
        tmp_clean = np.zeros(self._cube_images.shape)
        for idx in range(self._Nframes):
            tmp_clean[:,:,idx]  = self._cube_images[:,:,idx] - self._master_background - self._master_dark
        self._clean_images = tmp_clean
    
    @staticmethod    
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        clean_images = hduList[0].data 
        z_coeff = hduList[1].data 
        Nframes = header['N_FR']
        texp = header['T_EX_MS']
        return texp, Nframes, clean_images, z_coeff
    
class AnalyzeCleanImages():
    
    def __init__(self, fname):
        self._texp, self._Nframes,\
         self._clean_images, self._z_coeff = MeasureCleanImages.load(fname)
        
    def _compute_temporal_mean_from_cube(self):
        self._mean_ima = self._clean_images.mean(axis = 2)
        
    def _compute_temporal_std_from_cube(self):
        self._std_ima = self._clean_images.std(axis = 2)
        
    def compute_temporal_mean_and_std_from_cube(self):
        self._compute_temporal_mean_from_cube()
        self._compute_temporal_std_from_cube()
        
    def _gaussian_fit(self, image, err_im, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
        dimy, dimx = image.shape
        y, x = np.mgrid[:dimy, :dimx]
        fitter = LevMarLSQFitter(calc_uncertainties=True)
        model = Gaussian2D(amplitude=amplitude,
                           x_mean=x_mean, y_mean=y_mean,
                           x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                           y_stddev=fwhm_y * gaussian_fwhm_to_sigma)
        w = 1/err_im
        fit = fitter(model, x, y, z = image)
        return fit
    
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_max(self, image, ymax, xmax, size):
        a = size // 2
        cut_image = image[ymax-a:ymax+a+1, xmax-a:xmax+a+1]
        return cut_image
    
    def execute_Gaussianfitting(self):
        imax, ymax, xmax = self._get_image_peak_and_coords(self._mean_ima)
        imm = self._cut_image_around_max(self._mean_ima, ymax, xmax, 50)
        imax, ymax_cut, xmax_cut = self._get_image_peak_and_coords(imm)
        #image = self._cut_image_around_max(self._mean_ima, ymax, xmax)
        #err_ima = self._cut_image_around_max(self._std_ima, ymax, xmax)
        #image = self._mean_ima
        #image = imm
        #err_ima = self._std_ima
        err_ima = self._cut_image_around_max(self._std_ima, ymax, xmax, 50)
        # assert imm.shape == err_ima.shape
        fit = self._gaussian_fit(imm, err_ima, xmax_cut, ymax_cut, 3.3, 3.3, imax)
        self._fit = fit
        fit.cov_matrix
        par = fit.parameters
        err = np.sqrt(np.diag(fit.cov_matrix.cov_matrix))
        # ricorda di convertire da sigma a FWHM su x e y
        par[3] = par[3]/gaussian_fwhm_to_sigma 
        err[3] = err[3]/gaussian_fwhm_to_sigma 
        par[4] = par[4]/gaussian_fwhm_to_sigma
        err[4] = err[4]/gaussian_fwhm_to_sigma 
        self._par = par
        self._err = err
        print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y')
        print(par)
        print(err)
        return par, err
    
    def show_psf_profile(self):
        import matplotlib.pyplot as plt
        imax, yc, xc = self._get_image_peak_and_coords(self._mean_ima)
        size = 24
        data_x = np.arange(size+1)
        image = self._cut_image_around_max(self._mean_ima, yc, xc, size)
        err_ima = self._cut_image_around_max(self._std_ima, yc, xc, size)
        plt.figure()
        plt.title('PSF profile')
        plt.hlines(self._par[0], 0, size, ls='--', color='black', label='$A_{fit}$')
        plt.hlines(self._par[0]*0.5, 0, size, ls='--', color='black')
        #plt.hlines(self._par[0]*0.5, 0, size, ls='--', color='b', label='$FWHM$')
        plt.plot(image[size//2,:], 'ro', label='along x')
        plt.errorbar(data_x, image[size//2,:] , err_ima[size//2,:], ls=None,
                         fmt='.', markersize = 0.5, label='$\sigma_x$')
        plt.plot(image[:,size//2], 'bo', label='along y')
        
        plt.errorbar(data_x, image[:,size//2] , err_ima[:,size//2], ls=None,
                         fmt='.', markersize = 0.5, label='$\sigma_y$')
        plt.legend(loc='best')
        plt.grid(ls='--', alpha=0.5)
        
       
        
        plt.figure()
        plt.title('Clean PSF')
        plt.imshow(self._mean_ima, vmax=self._mean_ima.max(),\
                    vmin = self._mean_ima.min(), cmap = 'jet')
        plt.colorbar()
        