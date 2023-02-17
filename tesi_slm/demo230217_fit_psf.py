'''
Created on 17 Feb 2023

@author: labot
'''
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D


def gaussian_fit(image, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
    dimy, dimx = image.shape
    y, x = np.mgrid[:dimy, :dimx]
    fitter = LevMarLSQFitter(calc_uncertainties=True)
    model = Gaussian2D(amplitude=amplitude,
                       x_mean=x_mean, y_mean=y_mean,
                       x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                       y_stddev=fwhm_y * gaussian_fwhm_to_sigma)
    fit = fitter(model, x, y, image)
    return fit
    