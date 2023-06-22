import numpy as np 
from tesi_slm.tilt_linearity_on_camera import TiltedPsfMeasurer
from tesi_slm.my_tools import cut_image_around_coord, \
    get_index_from_image, execute_gaussian_fit_on_image, get_index_from_array
from scipy.optimize import curve_fit
from astropy.units import pix

class TiltedPsfAnalyzer():
    
    def __init__(self, fname):
        self._tilts_cube, self._c_span, self._Nframes, self._texp,\
         self._j_noll, self._init_coeff = TiltedPsfMeasurer.load_measures(fname)
         
    def compute_tilted_psf_desplacement(self, diffraction_limit_fwhm = 3.26, half_side_roi=25):
        
        num_of_tilts = self._tilts_cube.shape[0]
        
        self._pos_x = np.zeros(num_of_tilts)
        self._pos_y = np.zeros(num_of_tilts)
        self._amplitude = np.zeros(num_of_tilts)
        self._fwhm_x = np.zeros(num_of_tilts)
        self._fwhm_y = np.zeros(num_of_tilts)
        self._theta = np.zeros(num_of_tilts)
        self._pos_x_err = np.zeros(num_of_tilts)
        self._pos_y_err = np.zeros(num_of_tilts)
        self._amplitude_err = np.zeros(num_of_tilts)
        self._fwhm_x_err = np.zeros(num_of_tilts)
        self._fwhm_y_err = np.zeros(num_of_tilts)
        self._theta_err = np.zeros(num_of_tilts)
        
        for idx in range(num_of_tilts):
            
            tilt_ima = self._tilts_cube[idx] 
            y,x = get_index_from_image(tilt_ima, tilt_ima.max())
            half_side = half_side_roi
            roi_tilt = cut_image_around_coord(tilt_ima, y, x, half_side)
            
            fit_par, fit_err = execute_gaussian_fit_on_image(
                roi_tilt, diffraction_limit_fwhm, diffraction_limit_fwhm, False)
            
            self._amplitude[idx], self._pos_x[idx],  self._pos_y[idx], \
                self._fwhm_x[idx], self._fwhm_y[idx],  self._theta[idx] = fit_par
                
            self._amplitude_err[idx], self._pos_x_err[idx], \
                self._pos_y_err[idx], self._fwhm_x_err[idx],\
                    self._fwhm_y_err[idx], self._theta_err[idx] = fit_err
            
            self._pos_x[idx] = x - half_side + self._pos_x[idx]
            self._pos_y[idx] = y - half_side + self._pos_y[idx]
            
    def show_fitted_parametes_as_a_function_of_applyed_tilts(self, diffraction_limit_fwhm = None):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(self._c_span, self._fwhm_x, 'bo-', label='FWHM-X')
        plt.errorbar(self._c_span, self._fwhm_x, self._fwhm_x_err, fmt='.', ecolor ='b')
        plt.plot(self._c_span, self._fwhm_y, 'ro-', label='FWHM-Y')
        plt.errorbar(self._c_span, self._fwhm_y, self._fwhm_y_err, fmt='.', ecolor ='r')
        if diffraction_limit_fwhm is not None:
            plt.hlines(diffraction_limit_fwhm, self._c_span.min(),\
                        self._c_span.max(), 'k', '--', label = 'Diffraction limit')
            
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('FWHM [pixel]')
        
        plt.figure()
        plt.plot(self._c_span, self._amplitude, 'ro-', )
        plt.errorbar(self._c_span, self._amplitude, self._amplitude_err, fmt='.', ecolor ='r')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('Amplitude [ADU]')
    
    def show_tilt_desplacement(self):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(self._c_span, self._pos_x,'bo-')
        plt.errorbar(self._c_span, self._pos_x, self._pos_x_err, fmt='.', ecolor='b')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('X position [pixel]')
        
        plt.figure()
        plt.plot(self._c_span, self._pos_y,'ro-')
        plt.errorbar(self._c_span, self._pos_y, self._pos_y_err, fmt='.', ecolor='r')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('Y position [pixel]')
        
        parX, covX = np.polyfit(
            self._normalize(self._c_span), self._normalize(self._pos_x), 1, cov=True)  
        errX = np.sqrt(covX.diagonal())
        
        parY, covY = np.polyfit(
            self._normalize(self._c_span), self._normalize(self._pos_y), 1, cov=True)  
        errY = np.sqrt(covY.diagonal())
        
        plt.figure()
        plt.plot(self._normalize(self._c_span),self._normalize(self._pos_x),'bo-',label='along-x')
        plt.plot(self._normalize(self._c_span),self._normalize(self._pos_y),'ro-',label='along-y')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [normalized]$'%self._j_noll)
        plt.ylabel('Normalized position')
        plt.legend()
        
        
        return parX,errX,parY,errY
    
    def _normalize(self, linear_vector):
        a = linear_vector.min()
        b = linear_vector.max()
        x = (linear_vector - a)/(b-a)
        return x
    
    def execute_curve_fit(self):
        y0, x0 = self._pos_y[self._c_span == 0], self._pos_x[self._c_span == 0]
        d = np.sqrt((self._pos_x-x0)**2+(self._pos_y-y0)**2) 
        popt, pcov = curve_fit(
            self._func, 
            self._c_span + self._init_coeff[0],
            d,
            p0=[250e-3, 10.7e-3, 4.65e-6, 0],
            bounds=([200e-3, 10.5e-3, 0, 0], [250e-3, 10.7e-3, 4.65e-6, d.max()]),
            )
        return popt,pcov
    
    def _func(self, x , f, Dpe, pixel_pitch, offset):
        return 4*f*x/(Dpe*pixel_pitch) + offset
    
    def execite_linfit_along1axis(self):
        y0, x0 = self._pos_y[self._c_span == 0], self._pos_x[self._c_span == 0]
        c_span = self._c_span #np.delete(self._c_span, np.where(self._c_span == 0)[0][0])
        #print(c_span)
        if self._j_noll == 2 :
            pos = self._pos_x #np.delete(self._pos_x, np.where(self._c_span == 0)[0][0])
            err_pos = self._pos_x_err #np.delete(self._pos_x_err, np.where(self._c_span == 0)[0][0])
            ref_pos = x0
            
        if self._j_noll == 3 :
            pos = self._pos_y #np.delete(self._pos_y, np.where(self._c_span == 0)[0][0])
            err_pos = self._pos_y_err #np.delete(self._pos_y_err, np.where(self._c_span == 0)[0][0])
            ref_pos = y0
            
        obs_displacement = pos - ref_pos    
        coeff, cov= np.polyfit(c_span, obs_displacement, 1, cov=True, full=False)
        a,b = coeff
        err_coeff = np.sqrt(cov.diagonal())
        
        sigma_a = err_coeff[0]
        fit_displacement = a * c_span + b
        
        exp_displacement = 4*c_span *250e-3/(10.7e-3*4.65e-6)
        a_exp, b_exp = np.polyfit(c_span, -exp_displacement, 1)
        #linearity goodness estimations
        #slope ratio
        slope_ratio = a/a_exp
        err_slope = np.sqrt((sigma_a/a_exp)**2)
        #print(np.sqrt(((a_exp/a**2)*sigma_a)**2))
        #R-squared
        R2 = np.sum((fit_displacement + exp_displacement)**2)/np.sum((obs_displacement+exp_displacement)**2)
        #print(1-(np.sum((obs_displacement-fit_displacement)**2)/np.sum((obs_displacement+exp_displacement)**2)))
        print('R**2 = %g '%R2)
        
        chisq = np.sum(((obs_displacement - fit_displacement)**2/(err_pos)**2))
        redchi = chisq/(len(self._c_span)-1)
        rms_diff_obs_minus_fit = (obs_displacement - fit_displacement).std()
        
        import matplotlib.pyplot as plt
        pix_um = 4.65
        scale = 1e-6 
        plt.subplots(2, 1, sharex=True)
        plt.subplot(2,1,1)
        plt.plot(c_span/scale, obs_displacement*pix_um, 'x',markersize=7, label='measured')
        plt.plot(c_span/scale, fit_displacement*pix_um, 'r-', label='fit')
        plt.plot(c_span/scale, - exp_displacement*pix_um, 'k--',label='expected')
        plt.xlabel('$c_{%d} [\mu m]$ rms' %self._j_noll)
        plt.ylabel('Displacement $[\mu m]$')
        plt.legend(loc = 'best')
        plt.grid('--', alpha = 0.3)
        plt.subplot(2,1,2)
        plt.plot(c_span/scale, (obs_displacement + exp_displacement)*pix_um,'xb', label='meas - exp')
        plt.plot(c_span/scale, (fit_displacement + exp_displacement)*pix_um,'xr' ,label='fit - exp')
        plt.plot(c_span/scale, (obs_displacement - fit_displacement)*pix_um,'xk' ,label='meas - fit')
        plt.legend(loc='best')
        plt.ylabel('Difference $[\mu m]$')
        plt.xlabel('$c_{%d} [\mu m]$ rms' %self._j_noll)
        plt.grid('--', alpha = 0.3)
                
        return coeff,err_coeff, redchi, rms_diff_obs_minus_fit, slope_ratio, err_slope, R2
        
        
        
        
         
    
    # def show_tilt_desplacement(self, f = 250e-3, Dpe = 10.7e-3, pixel_size = 4.65e-6):
    #     sag = 4 * self._c_span
    #     alpha = sag / Dpe
    #
    #     if self._j_noll == 2:
    #         self._expecetd_psf_deltaX = f * alpha / pixel_size
    #         self._expecetd_psf_deltaY = np.zeros_like(
    #             self._expecetd_psf_deltaX)
    #     if self._j_noll == 3:
    #         self._expecetd_psf_deltaY = f * alpha / pixel_size
    #         self._expecetd_psf_deltaX = np.zeros_like(
    #             self._expecetd_psf_deltaY)
    #
    #     coeff_zero_idx = get_index_from_array(self._c_span, 0.)
    #     self._observed_psf_deltaX = self._pos_x[coeff_zero_idx]- self._pos_x     
    #     self._observed_psf_deltaY = self._pos_y[coeff_zero_idx]- self._pos_y
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure()
    #     plt.title('PSF displacement along x-axis')
    #     plt.plot(self._c_span, self._expecetd_psf_deltaX,
    #              'ro', label='expected')
    #     plt.plot(self._c_span, self._observed_psf_deltaX,
    #              'bx', label='measured')
    #     plt.errorbar(self._c_span, self._observed_psf_deltaX, self._pos_x_err, ls=None,
    #                  fmt='.', markersize=0.5, label='$\sigma$')
    #     plt.xlabel('$c_{%d}[m]$' % self._j_noll)
    #     plt.ylabel('$\Delta X_{pixels}$')
    #     plt.grid('--', alpha=0.3)
    #     plt.legend(loc='best')
    #
    #     plt.figure()
    #     plt.title('PSF displacement along y-axis')
    #     plt.plot(self._c_span, self._expecetd_psf_deltaY,
    #              'r--', label='expected')
    #     plt.plot(self._c_span, self._observed_psf_deltaY,
    #              'bx', label='measured')
    #     plt.errorbar(self._c_span, self._observed_psf_deltaY, self._pos_y_err, ls=None,
    #                  fmt='.', markersize=0.5, label='$\sigma$')
    #     plt.xlabel('$c_{%d}[m]$' % self._j_noll)
    #     plt.ylabel('$\Delta Y_{pixels}$')
    #     plt.grid('--', alpha=0.3)
    #     plt.legend()
    #
    #     plt.figure()
    #     plt.title('Residual displacement along y-axis')
    #     plt.plot(self._c_span, self._observed_psf_deltaY - self._expecetd_psf_deltaY,
    #              'ro')
    #     plt.errorbar(self._c_span, self._observed_psf_deltaY - self._expecetd_psf_deltaY, self._pos_y_err, ls=None,
    #                  fmt='.', markersize=0.5, label='$\sigma_y$')
    #     plt.xlabel('$c_{%d}[m]$' % self._j_noll)
    #     plt.ylabel('$\Delta Y_{pixels}$')
    #     plt.grid('--', alpha=0.3)
    #     plt.legend()
    #
    #     plt.figure()
    #     plt.title('Residual displacement along x-axis')
    #     plt.plot(self._c_span, self._observed_psf_deltaX - self._expecetd_psf_deltaX,
    #              'ro')
    #     plt.errorbar(self._c_span, self._observed_psf_deltaX - self._expecetd_psf_deltaX, self._pos_x_err, ls=None,
    #                  fmt='.', markersize=0.5, label='$\sigma_x$')
    #     plt.xlabel('$c_{%d}[m]$' % self._j_noll)
    #     plt.ylabel('$\Delta X_{pixels}$')
    #     plt.grid('--', alpha=0.3)
    #     plt.legend()
    #
