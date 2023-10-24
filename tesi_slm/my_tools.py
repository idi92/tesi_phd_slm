import numpy as np 
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator



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

def get_clean_cube_images(cube_image, master_dark, master_background):
    Nframes = cube_image.shape[-1]
    clean_cube = np.zeros(cube_image.shape)
    for n in range(Nframes):
        image = cube_image[:, :, n]
        clean_cube[:, :, n] = clean_image(image, master_dark, master_background)
    return clean_cube

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

def convert_uint8_2wrapped_phase(uint8_array, phase_wrap = 2*np.pi):
    
    unit = phase_wrap / 256
    # se restituivo
    # unit * unit8_array
    # nei valori a True avevo unit mentre lo voglio a zero
    wrapped_phase = \
    np.ma.array(data = unit * uint8_array.data, mask = uint8_array.mask, dtype=float)
    return wrapped_phase

def convert_opd2wrapped_phase(wf, wl = 635e-9, phase_wrap = np.pi*2):
    wf_uint8 = convert2uint8(wf, wl)
    wrapped_phase = convert_uint8_2wrapped_phase(wf_uint8, phase_wrap)
    return wrapped_phase

def opd2phase(self, opd, wl = 635e-9):

    return 2 * np.pi * opd / wl

def phase2opd(self, phase, wl = 635e-9):

    return 0.5 * wl * phase / np.pi


def save_clean_psf(fname, ima_clean, texp, Nframes, par, err, coeff):
    from astropy.io import fits
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    hdr['N_AV_FR'] = Nframes
        
    fits.writeto(fname ,ima_clean, hdr)
    fits.append(fname, par)
    fits.append(fname, err)
    fits.append(fname, coeff)
    
def open_fits_file(fname):
    from astropy.io import fits
    header = fits.getheader(fname)
    hduList = fits.open(fname)
    return header, hduList

def get_circular_mask_obj(centerYX = (571, 875), RadiusInPixel = 571, frameshape=(1152, 1920)):
    cmask = CircularMask(
        frameShape = frameshape,
        maskRadius = RadiusInPixel,
        maskCenter = centerYX)
    return cmask

def get_circular_mask(centerYX = (571, 875), RadiusInPixel = 571, frameshape=(1152, 1920)):
    return get_circular_mask_obj(centerYX, RadiusInPixel, frameshape).mask()

def get_wf_as_zerike_combo(cmask_obj, zernike_coefficients_in_meters):
    zernike_builder = ZernikeGenerator(cmask_obj)
    image_to_display = np.zeros((1152,1920))
    image_to_display = np.ma.array(data = image_to_display, mask = cmask_obj.mask(), fill_value = 0)
    for j, aj in enumerate(zernike_coefficients_in_meters):
        Zj = zernike_builder.getZernike(j + 2)
        image_to_display += aj * Zj
    return image_to_display

def get_zernike_wf(cmask_obj, j, aj = 1):
    zernike_builder = ZernikeGenerator(cmask_obj)
    image_to_display = np.zeros((1152,1920))
    image_to_display = np.ma.array(data = image_to_display, mask = cmask_obj.mask(), fill_value = 0)
    Zj = zernike_builder.getZernike(j)
    image_to_display = aj * Zj
    return image_to_display