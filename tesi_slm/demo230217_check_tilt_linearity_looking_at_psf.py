import numpy as np
from tesi_slm import psf_on_camera_optimizer
from tesi_slm import demo230217_fit_psf
from astropy.io import fits

def main():
    
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    # loading bg noise to subtract
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230220'
    fname = '/230220bg_camera.fits'
    fname_bg = fdir + fname
    poco.load_camera_background(fname_bg)
    poco.change_circular_mask(centerYX=(576,960), RadiusInPixel=555)
    texp = poco._texp_bg  #0.125ms
    Nframes = 100
    # in c2_span must be 0e-9
    Ntilts = 11
    c2_span = np.linspace(-2000e-9, 2000e-9, Ntilts)
    
    # expected tilt positions
    sag = 4*c2_span
    Dpe = 10.2e-3
    f = 250e-3
    alpha = sag/Dpe
    pixel_size = 4.65e-6
    expecetd_psf_deltaX = f * alpha /pixel_size
    expecetd_psf_deltaY = np.zeros(Ntilts)
    
    # writing on slm a almost good psf
    # zernike coefficients estimated as in
    # demo230215_manual_search_of_best_psf.py 
    coeff = [0, 0, -2.25e-07, -1e-08, 1.1e-07, -7e-08, 3e-08, 0, 0, -4.5e-08]
    poco._write_zernike_on_slm(
                             zernike_coefficients_in_meters = coeff,
                             add_wfc = True)
    cam.setExposureTime(texp)
    x_peaks = np.zeros(Ntilts)
    y_peaks = np.zeros(Ntilts)
    x_err = np.zeros(Ntilts)
    y_err = np.zeros(Ntilts)
    for idx, c2 in enumerate(c2_span):
        coeff[0] = c2
        poco._write_zernike_on_slm(
                zernike_coefficients_in_meters = coeff,
                add_wfc = True)
        image = poco.get_frames_from_camera(Nframes)
        image_mean, image_sigma = poco.get_mean_and_std_from_frames(image)
        ima_sub_bg = poco._subtract_background_from_image(image_mean)
            
        peak, err_peak, x_p, y_p = \
            poco._get_mean_peak_and_error_from_image(ima_sub_bg, image_sigma)
        fit_res = demo230217_fit_psf.gaussian_fit(
            ima_sub_bg, image_sigma, x_p, y_p, 3, 3, peak)
        
        x_peaks[idx],y_peaks[idx] = fit_res.x_mean.value, fit_res.y_mean.value
        sigmas = np.sqrt(np.diag(fit_res.cov_matrix.cov_matrix))
        x_err[idx] = sigmas[1]
        y_err[idx] = sigmas[2]
    idx_c = np.where(c2_span == 0)[0][0]
    # note: c2 < 0 generates dpos >0 on focal plane
    observed_psf_deltaX = x_peaks[idx_c]- x_peaks     
    observed_psf_deltaY = y_peaks[idx_c] - y_peaks
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('along x')
    plt.plot(c2_span, expecetd_psf_deltaX, 'ro', label = 'expected')
    plt.plot(c2_span, observed_psf_deltaX, 'bx', label = 'measured')
    plt.errorbar(c2_span, observed_psf_deltaX , x_err, ls=None,
                     fmt='.', markersize=0.5, label='$\sigma$')
    plt.xlabel('$c_2[m]$')
    plt.ylabel('$\delta_{pixels}$')
    plt.grid()
    plt.legend(loc = 'best')
    plt.figure()
    plt.title('along y')
    plt.plot(c2_span, expecetd_psf_deltaY, 'r--', label = 'expected')
    plt.plot(c2_span, observed_psf_deltaY, 'bx', label = 'measured')
    plt.errorbar(c2_span, observed_psf_deltaY , y_err, ls=None,
                     fmt='.', markersize = 0.5, label='$\sigma$')
    plt.xlabel('$c_2[m]$')
    plt.ylabel('$\delta_{pixels}$')
    plt.grid()
    plt.legend()
    poco.close_slm()
    
    tiltfname = fdir + '/230222_measured_tilts_with_weights.fits'
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    hdr['N_AV_FR'] = Nframes
    fits.writeto(tiltfname, expecetd_psf_deltaX, hdr)
    fits.append(tiltfname, observed_psf_deltaX)
    fits.append(tiltfname, x_err)
    fits.append(tiltfname, expecetd_psf_deltaY)
    fits.append(tiltfname, observed_psf_deltaY)
    fits.append(tiltfname, y_err)
    fits.append(tiltfname, c2_span)
    
    return  observed_psf_deltaX, observed_psf_deltaY, expecetd_psf_deltaX, expecetd_psf_deltaY, c2_span

def load_tilted_psf_data(fname):
    header = fits.getheader(fname)
    hduList = fits.open(fname)
    exp_dx = hduList[0].data
    obs_dx = hduList[1].data
    err_x = hduList[2].data
    exp_dy = hduList[3].data
    obs_dy = hduList[4].data
    err_y = hduList[5].data
    c2_span = hduList[6].data
    Nframes = header['N_AV_FR']
    texp = header['T_EX_MS']
    return exp_dx, obs_dx, err_x, exp_dy, obs_dy, err_y, Nframes, texp