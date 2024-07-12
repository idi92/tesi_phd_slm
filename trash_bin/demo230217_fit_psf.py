
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D



def gaussian_fit(image, err_im, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
    dimy, dimx = image.shape
    y, x = np.mgrid[:dimy, :dimx]
    fitter = LevMarLSQFitter(calc_uncertainties=True)
    model = Gaussian2D(amplitude=amplitude,
                       x_mean=x_mean, y_mean=y_mean,
                       x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                       y_stddev=fwhm_y * gaussian_fwhm_to_sigma)
    fit = fitter(model, x, y, z = image, weights = 1/err_im)
    return fit, model

def main():
    from tesi_slm import demo230217_measure_and_save_psf
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230221'
    psf_fname = '/230221psf_itercoeffsearch_bg_sub_0125ms_1000frames.fits'
    psf_data = demo230217_measure_and_save_psf.load_psf(fdir+psf_fname)
    
    psf = psf_data[0]
    err_psf = psf_data[1]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(psf, cmap='jet');plt.colorbar();
    
    fit = gaussian_fit(
        image = psf,
        err_im= err_psf,
        x_mean = 671,
        y_mean = 504,
        fwhm_x = 3,
        fwhm_y = 3,
        amplitude = psf.max())
    fit
    fit.cov_matrix
    par = fit.parameters
    err = np.sqrt(np.diag(fit.cov_matrix.cov_matrix))
    # ricorda di convertire da sigma a FWHM su x e y
    par[3] = par[3]/gaussian_fwhm_to_sigma
    err[3] = err[3]/gaussian_fwhm_to_sigma
    par[3] = par[4]/gaussian_fwhm_to_sigma
    err[4] = err[4]/gaussian_fwhm_to_sigma 
    print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y')
    print(par)
    print(err)
    
    return psf_data, err_psf, fit