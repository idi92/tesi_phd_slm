import numpy as np
from tesi_slm.tilt_linearity_on_camera import TiltedPsfMeasurer
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D
from astropy.io import fits
#from scipy.optimize import curve_fit

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
    
    def _weighted_gaussian_fit(self, image, err_ima, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
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
    
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_max(self, image, ymax, xmax):
        cut_image = image[ymax-2:ymax+3, xmax-2:xmax+3]
        return cut_image
    
    def compute_tilted_psf_displacement(self, fwhm_x = 3, fwhm_y = 3, method ='fbyf'):
        
        Ntilts = len(self._c_span)
        Nframes = self._Nframes
        self._y_mean = np.zeros(Ntilts)
        self._x_mean = np.zeros(Ntilts)
        self._err_x = np.zeros(Ntilts)
        self._err_y = np.zeros(Ntilts)
        
        if(method=='fbyf'):
            for idx in range(Ntilts):
                
                yk = np.zeros(Nframes)
                xk = np.zeros(Nframes)
                print('tilt N%d'%idx)
                for k_idx in range(Nframes):
                    print('fitting frame N%d'%k_idx)
                    image = self._images_4d[idx, :, :, k_idx]
                    peak, yc, xc = self._get_image_peak_and_coords(image)
                    cut_image = self._cut_image_around_max(image, yc, xc)
                    
                    # center of the image_cut (3x3) [1,1]
                    fit = self._gaussian_fit(cut_image, 1, 1, fwhm_x, fwhm_y, peak)
                    par = fit.parameters
                    xk[k_idx] = par[1] + xc
                    yk[k_idx] = par[2] + yc
                    
                self._y_mean[idx] = yk.mean()
                self._err_y[idx] = yk.std()
                self._x_mean[idx] = xk.mean()
                self._err_x[idx] = xk.std()
                
        if(method=='collapse'):
            collapsed_images = self._images_4d.mean(axis=3)
            collapsed_errima = self._images_4d.std(axis=3)
            for idx in range(Ntilts):
                image = collapsed_images[idx]
                peak, yc, xc = self._get_image_peak_and_coords(image)
                cut_image = self._cut_image_around_max(image, yc, xc)  
                # center of the image_cut (3x3) [1,1]
                err_ima = collapsed_errima[idx, yc-2:yc+3, xc-2:xc+3] 
                fit = self._weighted_gaussian_fit(cut_image, err_ima, 1, 1, fwhm_x, fwhm_y, peak)
                par = fit.parameters
                #print(fit.parameters)
                #print(fit.cov_matrix)
                err = np.sqrt(np.diag(fit.cov_matrix.cov_matrix))
                self._y_mean[idx] = par[2] + yc
                self._err_y[idx] = err[2]
                self._x_mean[idx] = par[1] + xc
                self._err_x[idx] = err[1]
            
        idx_c = np.where(self._c_span == 0)[0][0]
        # note: c2 < 0 generates dpos >0 on focal plane
        self._observed_psf_deltaX = self._x_mean[idx_c]- self._x_mean     
        self._observed_psf_deltaY = self._y_mean[idx_c] - self._y_mean
        
        
    def show_linearity_plots(self, f = 250e-3, Dpe = 10.2e-3):
        sag = 4*self._c_span
        #Dpe = 10.2e-3
        #f = 250e-3
        alpha = sag/Dpe
        pixel_size = 4.65e-6
        if self._j_noll == 2 :
            self._expecetd_psf_deltaX = f * alpha /pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(self._expecetd_psf_deltaX)
        if self._j_noll == 3 :
            self._expecetd_psf_deltaY = f * alpha /pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(self._expecetd_psf_deltaY)
            
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('PSF displacement along x-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaX, 'ro', label = 'expected')
        plt.plot(self._c_span, self._observed_psf_deltaX, 'bx', label = 'measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaX , self._err_x, ls=None,
                         fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$'%self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend(loc = 'best')
        plt.figure()
        plt.title('PSF displacement along y-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaY, 'r--', label = 'expected')
        plt.plot(self._c_span, self._observed_psf_deltaY, 'bx', label = 'measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaY , self._err_y, ls=None,
                         fmt='.', markersize = 0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$'%self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()
        
    def load_data4plot(self, fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        self._observed_psf_deltaX = hduList[0].data
        self._x_mean = hduList[1].data
        self._err_x = hduList[2].data
        self._observed_psf_deltaY = hduList[3].data
        self._y_mean = hduList[4].data
        self._err_y = hduList[5].data
        self._c_span = hduList[6].data
        self._init_coeff = hduList[7].data
        self._texp = header['T_EX_MS']
        self._j_noll = header['Z_J']    
        
    def save_data4plot(self, fname):
            
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['Z_J'] = self._j_noll
         
        fits.writeto(fname, self._observed_psf_deltaX, hdr)
        fits.append(fname, self._x_mean)
        fits.append(fname, self._err_x)
        fits.append(fname, self._observed_psf_deltaY)
        fits.append(fname, self._y_mean)
        fits.append(fname, self._err_y)
        fits.append(fname, self._c_span)
        fits.append(fname, self._init_coeff)
        
class TiltedPsfDisplacementFitter():
    
    def __init__(self, fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        self._observed_psf_deltaX = hduList[0].data
        self._x_mean = hduList[1].data
        self._err_x = hduList[2].data
        self._observed_psf_deltaY = hduList[3].data
        self._y_mean = hduList[4].data
        self._err_y = hduList[5].data
        self._c_span = hduList[6].data
        self._init_coeff = hduList[7].data
        self._texp = header['T_EX_MS']
        self._j_noll = header['Z_J']
        self._build_expected_displacements()
        
    def _build_expected_displacements(self, f = 250e-3, Dpe = 10.2e-3, pixel_size = 4.65e-6):
        sag = 4*self._c_span
        alpha = sag/Dpe
        
        if self._j_noll == 2 :
            self._expecetd_psf_deltaX = f * alpha /pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(self._expecetd_psf_deltaX)
        if self._j_noll == 3 :
            self._expecetd_psf_deltaY = f * alpha /pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(self._expecetd_psf_deltaY)
            
    def execute_linear_fit(self): 
        
        meas_d = np.sqrt(self._observed_psf_deltaX**2 + self._observed_psf_deltaY**2)
        exp_d = np.sqrt(self._expecetd_psf_deltaX**2 + self._expecetd_psf_deltaY**2)
        
        import matplotlib.pyplot as plt
        sign = np.ones_like(self._c_span)
        sign[self._c_span<0] = -1
        
        der_d_dx = self._observed_psf_deltaX/meas_d
        der_d_dy = self._observed_psf_deltaY/meas_d
        err_d = np.sqrt((der_d_dx**2) * (self._err_x**2) + (der_d_dy**2) * (self._err_y**2))
        plt.figure()
        plt.plot(self._c_span, sign*meas_d, 'xk', label ='meas')
        plt.plot(self._c_span, sign*exp_d, '--', label ='exp')
        plt.errorbar(self._c_span, sign*meas_d , err_d, ls=None,
                         fmt='.', markersize = 0.5, label='$\sigma$')
        
        x = self._c_span
        y = sign*meas_d
        w = 1/err_d
        
        par, cov = np.polyfit(x, y, 1 , full=False, w= None, cov=True)
        
        # def func(data, a, b):
        #     return a * data + b
        #
        # par, cov = curve_fit(func, x, y, par)
        
        c_fit = np.linspace(self._c_span.min(),self._c_span.max(),100)
        d_fit = par[0] * c_fit + par[1]
        plt.plot(c_fit, d_fit, '--', label ='fit')
        
        plt.xlabel('$c_{%d}[m]$'%self._j_noll)
        plt.ylabel('$Displacement_{pixels}$')
        plt.grid()
        plt.legend(loc = 'best')
        res = meas_d - (par[0] * self._c_span + par[1])
        chisq = sum((res / 1)**2)
        print('fit parameters:')
        print(par)
        print('sigma:')
        print(np.sqrt(np.diagonal(cov)))
        print('Chi^2 = %g' % chisq)
        red = chisq / (len(self._c_span) - 2)
        print('chi^2/dof = %g' % red)

        return par, cov
    
   
        