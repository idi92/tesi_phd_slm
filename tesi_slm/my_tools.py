import numpy as np 
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D


def cut_image_around_coord(image2D, yc, xc, halfside=25):
        cut_image = image2D[yc-halfside:yc+halfside, xc-halfside:xc+halfside]
        return cut_image
    
def get_index_from_image(image2D, value = None):
        #peak = image.max()
        if value is None:
            value = image2D.max()
        y, x = np.where(image2D==value)[0][0], np.where(image2D==value)[1][0]
        return y, x
    
def get_index_from_array(array1D, value = None):
    if value is None:
        value = array1D.max()
    idx = np.where(array1D == value)[0][0]
    return idx

def _gaussian_fit(image, err_im, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
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
    
def execute_gaussian_fit_on_image(cut_image, FWHMx, FWHMy, print_par=True):
        ymax, xmax = get_index_from_image(cut_image)
        imax = cut_image.max()
        err_ima = 1 #self._cut_image_around_max(self._std_ima, ymax, xmax, 50)
        # assert imm.shape == err_ima.shape
        fit = _gaussian_fit(cut_image, err_ima, xmax, ymax, FWHMx, FWHMy, imax)
        
        fit.cov_matrix
        par = fit.parameters
        err = np.sqrt(np.diag(fit.cov_matrix.cov_matrix))
        # ricorda di convertire da sigma a FWHM su x e y
        par[3] = par[3]/gaussian_fwhm_to_sigma 
        err[3] = err[3]/gaussian_fwhm_to_sigma 
        par[4] = par[4]/gaussian_fwhm_to_sigma
        err[4] = err[4]/gaussian_fwhm_to_sigma
        if print_par is True: 
            print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y')
            print(par)
            print(err)
        return par, err
    
def clean_image(image, master_dark, master_background):
    return image - master_background - master_dark

def clean_cube_images(cube_image, master_dark, master_background):
    Nframes = cube_image.shape[-1]
    clean_cube = np.zeros(cube_image.shape)
    for n in range(Nframes):
        image = cube_image[:, :, n]
        clean_cube[:, :, n] = clean_image(image, master_dark, master_background)
    return clean_cube.mean(axis = 2)

def reshape_map2vector(array2d, length = 2211840, method ='C'):
    return np.reshape(array2d, (length,), method)

def reshape_vector2map(vector, shape = [1152, 1920], method ='C'):
    return np.reshape(vector, (shape[0], shape[1]), method)

def convert2uint8(array, wrapping_val = 635e-9):
    '''
    wrapping_val is the calibration wl or the phase wrap value
    '''
    data = array * 256 / wrapping_val
    data = np.round(data)
    return data.astype(np.uint8)

def convert_uint8_2wrapped_phase(uint8_array):
    unit = 2 * np.pi / 256
    return unit * uint8_array
    
    