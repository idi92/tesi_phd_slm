from prove_su_slm import  psf_on_camera_optimizer
import numpy as np 
import pysilico

def main(): 
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    poco.set_slm_flat(add_wfc = True)
    initial_coeff = [ 0.,  0., -1.2e-07, 8e-8,  8e-08,  4.e-08,  4.e-08]
    #initial_coeff = [ 0.,  0., -1.2e-07]
    best_coeff = poco.compute_zernike_coeff2optimize_psf(
        list_of_starting_coeffs_in_meters = initial_coeff,
        max_amp = 200e-9,
        min_amp = -200e-9)
    poco.show_psf_comparison_wrt_slm_flat(
        z_coeff_list_in_meters = best_coeff,
        texp_in_ms = 0.125,
        Nframe2average = 30,
        add_wfc =True)
    poco.close_slm()
    
    return best_coeff