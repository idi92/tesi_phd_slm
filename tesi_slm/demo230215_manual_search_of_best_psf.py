from tesi_slm import psf_on_camera_optimizer
import numpy as np

def main():
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    best_coeff = [0, 0, -1.94e-7, 4.32e-8, 1.264e-7, -6.32e-8]
    init_coeff = best_coeff.copy()
    # Z4
    poco.search_zernike_coeff2optimize_psf(4, np.lispace(-200e-9, 200e-9, 11), best_coeff)
    # observed maximum at -1.6e-7
    best_coeff [4-2] = -1.6e-7
    # scrinking amp
    poco.search_zernike_coeff2optimize_psf(4, np.lispace(-1.65e-7, -1.55e-7, 11), best_coeff)
    #observed fluctuations
    best_coeff[4-2] = -1.6e-7
    poco.search_zernike_coeff2optimize_psf(5, np.lispace(-200e-9, 200e-9, 11), best_coeff)
    # to be continued
    best_coeff[5-2] = 0
    poco.close_slm() 