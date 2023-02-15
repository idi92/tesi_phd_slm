from prove_su_slm import  psf_on_camera_optimizer
import numpy as np 
import pysilico

def main():
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    poco.set_slm_flat(add_wfc = True)
    #these zernike coefficients are the ones that reproduce
    #an almost decent psf for the current (mis)alignment
    # TODO : put an x-y stage on the AC doublet and check the faces
    coeff = [0 , 0, -1.94e-7, 4.32e-8, 1.264e-7, -6.32e-8]
    poco._write_zernike_on_slm(
                             zernike_coefficients_in_meters = coeff,
                             add_wfc = True)
    c2_max = 1e-6
    c2_min = -1e-6
    Ntilts = 21
    c2_span = np.linspace(c2_min, c2_max, Ntilts)
    sag = 4*c2_span
    Dpe = 10.2e-3
    f = 250e-3
    alpha = sag/Dpe
    pixel_size = 4.65e-6
    expecetd_psf_deltaX = f * alpha /pixel_size
    expecetd_psf_deltaY = np.zeros(Ntilts)
    texp = 0.2
    cam.setExposureTime(texp)
    
    poco._write_zernike_on_slm(
                             zernike_coefficients_in_meters = coeff,
                             add_wfc = True)
    psf_ima = poco.get_image_from_camera(frame_to_avarage = 40)
    max, ycenter, xcenter = poco._get_a_better_estimate_of_peak_intensity(psf_ima)
    
    observed_psf_deltaX = np.zeros(Ntilts)
    observed_psf_deltaY = np.zeros(Ntilts)
    for k, c2 in enumerate(c2_span):
        coeff[0] = c2
        poco._write_zernike_on_slm(
                             zernike_coefficients_in_meters = coeff,
                             add_wfc = True)
        tilt_ima = poco.get_image_from_camera(frame_to_avarage = 40)
        max_tilt, yc, xc = poco._get_a_better_estimate_of_peak_intensity(tilt_ima)
        observed_psf_deltaX[k] = xcenter - xc     
        observed_psf_deltaY[k] = ycenter - yc
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('along x')
    plt.plot(c2_span, expecetd_psf_deltaX, 'ro', label = 'expected')
    plt.plot(c2_span, observed_psf_deltaX, 'bx', label = 'measured')
    plt.grid()
    plt.legend(loc = 'best')
    plt.figure()
    plt.title('along y')
    plt.plot(c2_span, expecetd_psf_deltaY, 'r--', label = 'expected')
    plt.plot(c2_span, observed_psf_deltaY, 'bx', label = 'measured')
    poco.close_slm()
    return  observed_psf_deltaX, observed_psf_deltaY, expecetd_psf_deltaX, expecetd_psf_deltaY



