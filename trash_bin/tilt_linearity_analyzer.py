import numpy as np
from tesi_slm.tilt_linearity_on_camera import TiltedPsfMeasurer
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D
from astropy.io import fits
#from scipy.optimize import curve_fit

class TiltedPsfAnalyzerTRASHME():
    
    def __init__(self, fname):
        
        self._images_3d, self._c_span, self._Nframes, self._texp,\
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
    
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_max(self, image, ymax, xmax):
        cut_image = image[ymax-2:ymax+3, xmax-2:xmax+3]
        return cut_image
    
    def compute_tilted_psf_displacement(self, fwhm_x = 3.3 , fwhm_y=3.3 , method ='fbyf'):
        
        Ntilts = len(self._c_span)
        Nframes = self._Nframes
        self._y_mean = np.zeros(Ntilts)
        self._x_mean = np.zeros(Ntilts)
        self._err_x = np.zeros(Ntilts)
        self._err_y = np.zeros(Ntilts)
        self._method = method
        
        if(method == 'fbyf'):
            print('method:%s'% method)
            for idx in range(Ntilts):
                
                yk = np.zeros(Nframes)
                xk = np.zeros(Nframes)
                print('tilt N%d'%idx)
                for k_idx in range(Nframes):
                    print('fitting frame N%d'%k_idx)
                    image = self._images_4d[idx, :, :, k_idx]
                    peak, yc, xc = self._get_image_peak_and_coords(image)
                    cut_image = self._cut_image_around_max(image, yc, xc)
                    
                    # center of the image_cut (5x5) [2,2]
                    fit = self._gaussian_fit(cut_image, 2, 2, fwhm_x, fwhm_y, peak)
                    par = fit.parameters
                    xk[k_idx] = par[1] + xc
                    yk[k_idx] = par[2] + yc
                    
                self._y_mean[idx] = yk.mean()
                self._err_y[idx] = yk.std()
                self._x_mean[idx] = xk.mean()
                self._err_x[idx] = xk.std()
                
        if(method =='collapse'):
            print('method:%s'%method)
            collapsed_images = self._images_4d.mean(axis=3)
            collapsed_errima = self._images_4d.std(axis=3)
            for idx in range(Ntilts):
                image = collapsed_images[idx]
                peak, yc, xc = self._get_image_peak_and_coords(image)
                err_ima = collapsed_errima[idx]
                fit = self._weighted_gaussian_fit(image, err_ima, xc, yc, fwhm_x, fwhm_y, peak)
                par = fit.parameters
                #print(fit.parameters)
                #print(fit.cov_matrix)
                err = np.sqrt(np.diag(fit.cov_matrix.cov_matrix))
                # conversion from sigma to FWHM 
                par[3] = par[3]/gaussian_fwhm_to_sigma
                err[3] = err[3]/gaussian_fwhm_to_sigma
                par[3] = par[4]/gaussian_fwhm_to_sigma
                err[4] = err[4]/gaussian_fwhm_to_sigma 
                print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y')
                print('tilted psf #%d'%idx)
                print(par)
                print(err)
                self._y_mean[idx] = par[2] 
                self._err_y[idx] = err[2]
                self._x_mean[idx] = par[1] 
                self._err_x[idx] = err[1]
                
        if(method == 'fbyfmax'):
            print('method:%s' % method)
            for idx in range(Ntilts):

                yk = np.zeros(Nframes)
                xk = np.zeros(Nframes)
                print('tilt N%d' % idx)
                for k_idx in range(Nframes):
                    print('Searching max in frame N%d' % k_idx)
                    image = self._images_4d[idx, :, :, k_idx]
                    peak, yk[k_idx], xk[k_idx] = self._get_image_peak_and_coords(
                        image)

                self._y_mean[idx] = yk.mean()
                self._err_y[idx] = yk.std()
                self._x_mean[idx] = xk.mean()
                self._err_x[idx] = xk.std()
            
        idx_c = np.where(self._c_span == 0)[0][0]
        # note: c2 < 0 generates dpos >0 on focal plane
        self._observed_psf_deltaX = self._x_mean[idx_c]- self._x_mean     
        self._observed_psf_deltaY = self._y_mean[idx_c] - self._y_mean
        
        
    def show_linearity_plots(self, f=250e-3, Dpe=10.2e-3):
        sag = 4 * self._c_span
        #Dpe = 10.2e-3
        #f = 250e-3
        alpha = sag / Dpe
        pixel_size = 4.65e-6
        if self._j_noll == 2:
            self._expecetd_psf_deltaX = f * alpha / pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(
                self._expecetd_psf_deltaX)
        if self._j_noll == 3:
            self._expecetd_psf_deltaY = f * alpha / pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(
                self._expecetd_psf_deltaY)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('PSF displacement along x-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaX,
                 'ro', label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaX,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend(loc='best')
        plt.figure()
        plt.title('PSF displacement along y-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaY,
                 'r--', label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaY,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()
        plt.figure()
        plt.title('Residual displacement along y-axis')
        plt.plot(self._c_span, self._observed_psf_deltaY - self._expecetd_psf_deltaY,
                 'ro', label=self._method)
        plt.errorbar(self._c_span, self._observed_psf_deltaY - self._expecetd_psf_deltaY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_y$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()
        plt.figure()
        plt.title('Residual displacement along x-axis')
        plt.plot(self._c_span, self._observed_psf_deltaX - self._expecetd_psf_deltaX,
                 'ro', label=self._method)
        plt.errorbar(self._c_span, self._observed_psf_deltaX - self._expecetd_psf_deltaX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_x$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend()
        
    @staticmethod
    def load_data4plot(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        observed_psf_deltaX = hduList[0].data
        x_mean = hduList[1].data
        err_x = hduList[2].data
        observed_psf_deltaY = hduList[3].data
        y_mean = hduList[4].data
        err_y = hduList[5].data
        c_span = hduList[6].data
        init_coeff = hduList[7].data
        texp = header['T_EX_MS']
        j_noll = header['Z_J']
        return observed_psf_deltaX, x_mean, err_x, observed_psf_deltaY, y_mean, err_y, \
            c_span, init_coeff, texp, j_noll
        
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
        
        
class ShowTiltedPsfTRASHME():

    def __init__(self, fname):
        self._observed_psf_deltaX, self._x_mean, self._err_x,\
            self._observed_psf_deltaY, self._y_mean, self._err_y,\
            self._c_span, self._init_coeff, self._texp, self._j_noll = TiltedPsfAnalyzer.load_data4plot(
                fname)

    def show_linearity_plots(self, f=250e-3, Dpe=10.2e-3, method='max'):
        self._method = method
        sag = 4 * self._c_span
        #Dpe = 10.2e-3
        #f = 250e-3
        alpha = sag / Dpe
        pixel_size = 4.65e-6
        if self._j_noll == 2:
            self._expecetd_psf_deltaX = f * alpha / pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(
                self._expecetd_psf_deltaX)
        if self._j_noll == 3:
            self._expecetd_psf_deltaY = f * alpha / pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(
                self._expecetd_psf_deltaY)
        self._resY = self._observed_psf_deltaY - self._expecetd_psf_deltaY
        self._resX = self._observed_psf_deltaX - self._expecetd_psf_deltaX

        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('PSF displacement along x-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaX,
                 'r--', lw=0.5, label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaX,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend(loc='best')
        plt.figure()
        plt.title('PSF displacement along y-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaY,
                 'r--', lw=0.5, label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaY,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()
        plt.figure()
        plt.title('Residual displacement along x-axis')
        plt.plot(self._c_span, self._resX, 'ko', label=self._method)
        plt.errorbar(self._c_span, self._resX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_x$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend()
        plt.figure()
        plt.title('Residual displacement along y-axis')
        plt.plot(self._c_span, self._resY, 'ko', label=self._method)
        plt.errorbar(self._c_span, self._resY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_y$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()

        return self._resX, self._resY, self._err_x, self._err_y, self._c_span

    def show_subplots(self,  f=250e-3, Dpe=10.2e-3, method='max'):
        self._method = method
        sag = 4 * self._c_span
        #Dpe = 10.2e-3
        #f = 250e-3
        alpha = sag / Dpe
        pixel_size = 4.65e-6
        if self._j_noll == 2:
            self._expecetd_psf_deltaX = f * alpha / pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(
                self._expecetd_psf_deltaX)
        if self._j_noll == 3:
            self._expecetd_psf_deltaY = f * alpha / pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(
                self._expecetd_psf_deltaY)
        self._resY = self._observed_psf_deltaY - self._expecetd_psf_deltaY
        self._resX = self._observed_psf_deltaX - self._expecetd_psf_deltaX

        import matplotlib.pyplot as plt
        plt.subplots(2, 1, sharex=True)
        plt.subplot(2, 1, 1)
        plt.title('PSF displacement along x-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaX,
                 'r--', lw=0.5, label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaX,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        plt.title('Residual displacement along x-axis')
        plt.plot(self._c_span, self._resX, 'ko', label=self._method)
        plt.errorbar(self._c_span, self._resX, self._err_x, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_x$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid()
        plt.legend()
        plt.subplots(2, 1, sharex=True)
        plt.subplot(2, 1, 1)
        plt.title('PSF displacement along y-axis')
        plt.plot(self._c_span, self._expecetd_psf_deltaY,
                 'r--', lw=0.5, label='expected')
        plt.plot(self._c_span, self._observed_psf_deltaY,
                 'bx', label='measured')
        plt.errorbar(self._c_span, self._observed_psf_deltaY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title('Residual displacement along y-axis')
        plt.plot(self._c_span, self._resY, 'ko', label=self._method)
        plt.errorbar(self._c_span, self._resY, self._err_y, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma_y$')
        plt.xlabel('$c_{%d}[m]$' % self._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid()
        plt.legend()

    def get_data(self, f=250e-3, Dpe=10.2e-3):
        sag = 4 * self._c_span

        #Dpe = 10.2e-3
        #f = 250e-3
        alpha = sag / Dpe
        pixel_size = 4.65e-6
        if self._j_noll == 2:
            self._expecetd_psf_deltaX = f * alpha / pixel_size
            self._expecetd_psf_deltaY = np.zeros_like(
                self._expecetd_psf_deltaX)
        if self._j_noll == 3:
            self._expecetd_psf_deltaY = f * alpha / pixel_size
            self._expecetd_psf_deltaX = np.zeros_like(
                self._expecetd_psf_deltaY)
        self._resY = self._observed_psf_deltaY - self._expecetd_psf_deltaY
        self._resX = self._observed_psf_deltaX - self._expecetd_psf_deltaX
        return self._resX, self._resY, self._err_x, self._err_y, self._c_span


class CompareDifferenceMethodsTRASHME():

    def __init__(self, fname_col, fname_max, fname_5x5):
        self._stp_col = ShowTiltedPsf(fname_col)
        self.data_col = self._stp_col.get_data()
        self._stp_max = ShowTiltedPsf(fname_max)
        self.data_max = self._stp_max.get_data()
        self._stp_5x5 = ShowTiltedPsf(fname_5x5)
        self.data_5x5 = self._stp_5x5.get_data()

    def show_difference(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Difference btw obs and exp along x axis')
        plt.plot(self.data_col[-1], self.data_col[0], 'kx', label='collapse')
        plt.errorbar(self.data_col[-1], self.data_col[0], self.data_col[2], ls=None,
                     fmt='.k')  # , markersize=0.5, color='k')
        plt.plot(self.data_max[-1], self.data_max[0], 'rx', label='max')
        plt.errorbar(self.data_max[-1], self.data_max[0], self.data_max[2], ls=None,
                     fmt='o', markersize=0.5, color='r')
        plt.plot(self.data_5x5[-1], self.data_5x5[0], 'bx', label='5x5')
        plt.errorbar(self.data_5x5[-1], self.data_5x5[0], self.data_5x5[2], ls=None,
                     fmt='o', markersize=0.5, color='b')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')

        plt.figure()
        plt.title('Difference btw obs and exp along y axis')
        plt.plot(self.data_col[-1], self.data_col[1], 'kx', label='collapse')
        plt.errorbar(self.data_col[-1], self.data_col[1], self.data_col[3], ls=None,
                     fmt='o',  capsize=0.2, markersize=0.5, color='k')
        plt.plot(self.data_max[-1], self.data_max[1], 'rx', label='max')
        plt.errorbar(self.data_max[-1], self.data_max[1], self.data_max[3], ls=None,
                     fmt='o', capsize=0.2, markersize=0.5, color='r')
        plt.plot(self.data_5x5[-1], self.data_5x5[1], 'bx', label='5x5')
        plt.errorbar(self.data_5x5[-1], self.data_5x5[1], self.data_5x5[3], ls=None,
                     fmt='o', capsize=0.2, markersize=0.5, color='b')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')

    def show_clear(self):
        import matplotlib.pyplot as plt
        plt.subplots(2, 1, sharex=True)

        plt.subplot(2, 1, 1)
        plt.title('Difference btw obs and exp along x axis')
        plt.plot(self.data_col[-1], self.data_col[0], 'ko-', label='collapse')
        plt.plot(self.data_max[-1], self.data_max[0], 'ro-', label='max')
        plt.plot(self.data_5x5[-1], self.data_5x5[0], 'bo-', label='5x5')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\Delta X_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        plt.plot(self.data_col[-1], self.data_col[2], 'ko-', label='collapse')
        plt.plot(self.data_max[-1], self.data_max[2], 'ro-', label='max')
        plt.plot(self.data_5x5[-1], self.data_5x5[2], 'bo-', label='5x5')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\sigma X_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')

        plt.subplots(2, 1, sharex=True)
        plt.subplot(2, 1, 1)
        plt.title('Difference btw obs and exp along y axis')
        plt.plot(self.data_col[-1], self.data_col[1], 'ko-', label='collapse')
        plt.plot(self.data_max[-1], self.data_max[1], 'ro-', label='max')
        plt.plot(self.data_5x5[-1], self.data_5x5[1], 'bo-', label='5x5')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\Delta Y_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        plt.plot(self.data_col[-1], self.data_col[3], 'ko-', label='collapse')
        plt.plot(self.data_max[-1], self.data_max[3], 'ro-', label='max')
        plt.plot(self.data_5x5[-1], self.data_5x5[3], 'bo-', label='5x5')
        plt.xlabel('$c_{%d}[m]$' % self._stp_col._j_noll)
        plt.ylabel('$\sigma Y_{pixels}$')
        plt.grid(ls='--', alpha=0.8)
        plt.legend(loc='best')

    def compute_res(self):
        var_colx = self.data_col[0].std()
        var_coly = self.data_col[1].std()
        var_maxx = self.data_max[0].std()
        var_maxy = self.data_max[1].std()
        var_5x5x = self.data_5x5[0].std()
        var_5x5y = self.data_5x5[1].std()
        return var_colx, var_coly, var_maxx, var_maxy, var_5x5x, var_5x5y        

#
# class TiltedPsfDisplacementFitter():
#
#     def __init__(self, fname):
#         header = fits.getheader(fname)
#         hduList = fits.open(fname)
#         self._observed_psf_deltaX = hduList[0].data
#         self._x_mean = hduList[1].data
#         self._err_x = hduList[2].data
#         self._observed_psf_deltaY = hduList[3].data
#         self._y_mean = hduList[4].data
#         self._err_y = hduList[5].data
#         self._c_span = hduList[6].data
#         self._init_coeff = hduList[7].data
#         self._texp = header['T_EX_MS']
#         self._j_noll = header['Z_J']
#         self._build_expected_displacements()
#
#     def _build_expected_displacements(self, f = 250e-3, Dpe = 10.2e-3, pixel_size = 4.65e-6):
#         sag = 4*self._c_span
#         alpha = sag/Dpe
#
#         if self._j_noll == 2 :
#             self._expecetd_psf_deltaX = f * alpha /pixel_size
#             self._expecetd_psf_deltaY = np.zeros_like(self._expecetd_psf_deltaX)
#         if self._j_noll == 3 :
#             self._expecetd_psf_deltaY = f * alpha /pixel_size
#             self._expecetd_psf_deltaX = np.zeros_like(self._expecetd_psf_deltaY)
#
#     def execute_linear_fit(self): 
#
#         meas_d = np.sqrt(self._observed_psf_deltaX**2 + self._observed_psf_deltaY**2)
#         exp_d = np.sqrt(self._expecetd_psf_deltaX**2 + self._expecetd_psf_deltaY**2)
#
#         import matplotlib.pyplot as plt
#         sign = np.ones_like(self._c_span)
#         sign[self._c_span<0] = -1
#
#         der_d_dx = self._observed_psf_deltaX/meas_d
#         der_d_dy = self._observed_psf_deltaY/meas_d
#         err_d = np.sqrt((der_d_dx**2) * (self._err_x**2) + (der_d_dy**2) * (self._err_y**2))
#         plt.figure()
#         plt.plot(self._c_span, sign*meas_d, 'xk', label ='meas')
#         plt.plot(self._c_span, sign*exp_d, '--', label ='exp')
#         plt.errorbar(self._c_span, sign*meas_d , err_d, ls=None,
#                          fmt='.', markersize = 0.5, label='$\sigma$')
#
#         x = self._c_span
#         y = sign*meas_d
#         w = 1/err_d
#
#         par, cov = np.polyfit(x, y, 1 , full=False, w= None, cov=True)
#
#         # def func(data, a, b):
#         #     return a * data + b
#         #
#         # par, cov = curve_fit(func, x, y, par)
#
#         c_fit = np.linspace(self._c_span.min(),self._c_span.max(),100)
#         d_fit = par[0] * c_fit + par[1]
#         plt.plot(c_fit, d_fit, '--', label ='fit')
#
#         plt.xlabel('$c_{%d}[m]$'%self._j_noll)
#         plt.ylabel('$Displacement_{pixels}$')
#         plt.grid()
#         plt.legend(loc = 'best')
#         res = meas_d - (par[0] * self._c_span + par[1])
#         chisq = sum((res / 1)**2)
#         print('fit parameters:')
#         print(par)
#         print('sigma:')
#         print(np.sqrt(np.diagonal(cov)))
#         print('Chi^2 = %g' % chisq)
#         red = chisq / (len(self._c_span) - 2)
#         print('chi^2/dof = %g' % red)
#
#         return par, cov
#
#

        