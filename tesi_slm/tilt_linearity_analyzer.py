import numpy as np
from tesi_slm.tilt_linearity_on_camera import TiltedPsfMeasurer
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D
from astropy.io import fits

class TiltedPsfAnalyzer():
    
    def __init__(self, fname):
        
        self._images_4d, self._c_span, self._Nframes, self._texp,\
         self._j_noll, self._init_coeff = TiltedPsfMeasurer.load_measures(fname)
         
         
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
    
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def compute_tilted_psf_desplacement(self, fwhm_x = 3, fwhm_y = 3):
        
        Ntilts = len(self._c_span)
        Nframes = self._Nframes
        self._y_mean = np.zeros(Ntilts)
        self._x_mean = np.zeros(Ntilts)
        self._err_x = np.zeros(Ntilts)
        self._err_y = np.zeros(Ntilts)
        
        for idx in range(Ntilts):
            
            yk = np.zeros(Nframes)
            xk = np.zeros(Nframes)
            
            for k_idx in range(Nframes):
                
                image = self._images_4d[idx, :, :, k_idx]
                peak, yc, xc = self._get_image_peak_and_coords(image)
                fit = self._gaussian_fit(image, xc, yc, fwhm_x, fwhm_y, peak)
                par = fit.parameters
                xk[k_idx] = par[1]
                yk[k_idx] = par[2]
                
            self._y_mean[idx] = yk.mean()
            self._err_y[idx] = yk.std()
            self._x_mean[idx] = xk.mean()
            self._err_x[idx] = xk.std()
        
        idx_c = np.where(self._c_span == 0)[0][0]
        # note: c2 < 0 generates dpos >0 on focal plane
        self._observed_psf_deltaX = self._x_mean[idx_c]- self._x_mean     
        self._observed_psf_deltaY = self._y_mean[idx_c] - self._y_mean
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('along x')
        #plt.plot(self._c_span, self._expecetd_psf_deltaX, 'ro', label = 'expected')
        plt.plot(self._c_span, self._observed_psf_deltaX, 'bx', label = 'measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaX , self._err_x, ls=None,
                         fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_2[m]$')
        plt.ylabel('$\delta_{pixels}$')
        plt.grid()
        plt.legend(loc = 'best')
        plt.figure()
        plt.title('along y')
        #plt.plot(self._c_span, self._expecetd_psf_deltaY, 'r--', label = 'expected')
        plt.plot(self._c_span, self._observed_psf_deltaY, 'bx', label = 'measured')
        plt.errorbar(self._c2_span, self._observed_psf_deltaY , self._err_y, ls=None,
                         fmt='.', markersize = 0.5, label='$\sigma$')
        plt.xlabel('$c_2[m]$')
        plt.ylabel('$\delta_{pixels}$')
        plt.grid()
        plt.legend()
        
        
        def save_data_plot(self, fname):
            
            hdr = fits.Header()
            hdr['T_EX_MS'] = self._texp
            hdr['Z_J'] = self._j_noll_idx
             
            fits.writeto(fname, self._observed_psf_deltaX, hdr)
            fits.append(fname, self._x_mean)
            fits.append(fname, self._x_err)
            fits.append(fname, self._observed_psf_deltaY)
            fits.append(fname, self._y_mean)
            fits.append(fname, self._y_err)
            fits.append(fname, self._c_span)
            fits.append(fname,self._init_coeff)