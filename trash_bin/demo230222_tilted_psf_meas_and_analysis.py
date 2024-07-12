from tesi_slm.tilt_linearity_on_camera import TiltedPsfMeasurer
from tesi_slm.tilt_linearity_analyzer import TiltedPsfAnalyzer, \
 TiltedPsfDisplacementFitter
import numpy as np

def measure_example():
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230221'
    fname_tilt = '/230221_Z2tilted_psf_v2.fits'
    tmp = TiltedPsfMeasurer()
    # misuro Z2 con c2 tra -500 e 500 nm
    j = 2; c_span = np.linspace(-500e-9, 500e-9, 11); texp = 0.125
    init_coeff = [0, 0, -2.25e-07, -1e-08, 1.1e-07, -7e-08, 3e-08, 0, 0, -4.5e-08]
    tmp.measure_tilted_psf(j, c_span, texp, init_coeff)
    tmp.save_measures(fdir+fname_tilt)

def analyze_example():
    # carico le misure fatte su Z2 con c2 tra -500 e 500 nm
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230221'
    fname_tilt = '/230221_Z2tilted_psf_v2.fits'
    fname = fdir + fname_tilt
    
    tpa = TiltedPsfAnalyzer(fname)
    # FWHM of the measured psf
    tpa.compute_tilted_psf_displacement(fwhm_x=3, fwhm_y=3, method='fbyf')
    # error occurs when choosing method='collapse'
    # tpa.compute_tilted_psf_displacement(fwhm_x=3, fwhm_y=3, method='collapse')
    tpa.show_linearity_plots(f=250e-3, Dpe=10.2e-3)
    fsave =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230222\\230222tpa_4plot_230221z2v2.fits"
    tpa.save_data4plot(fsave)

def fitting_example():
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230222\\230222tpa_4plot_230221z2v2.fits"
    tpdf = TiltedPsfDisplacementFitter(fname)
    par, cov = tpdf.execute_linear_fit()