import numpy as np
from tesi_slm import psf_on_camera_optimizer
from tesi_slm import demo230217_fit_psf

def main():
    
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    # loading bg noise to subtract
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230217'
    fname = '/230217bg_camera.fits'
    fname_bg = fdir + fname
    poco.load_camera_background(fname_bg)
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
    coeff = [0, 0, -230e-9, -80e-9, 140e-9, -30e-9, -20e-9, 0, 0, -30e-9]
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
            ima_sub_bg, x_p, y_p, 4, 4, peak)
        
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
                     fmt='.', markersize=0.5, label='$\sigma$')
    plt.xlabel('$c_2[m]$')
    plt.ylabel('$\delta_{pixels}$')
    plt.grid()
    plt.legend()
    poco.close_slm()
    return  observed_psf_deltaX, observed_psf_deltaY, expecetd_psf_deltaX, expecetd_psf_deltaY