import numpy as np
from tesi_slm.fft4slm_diffraction_patterns import DiffractionModel1D
from tesi_slm import sharp_psf_on_camera, my_tools
import matplotlib.pyplot as plt

def tilt_fft():
    
    phase_wrap = 2 * np.pi
    M = phase_wrap / 2* np.pi
    dm1d = DiffractionModel1D()
    c2 = 500e-9
    spatial_period = dm1d.get_sawtooth_spatial_period_from_c2(c2, phase_wrap)
    
    Dpe = 10.5e-3
    # qui il teorema del campionamento
    #
    Nsamples = int(Dpe/1.e-6)
    xi = np.linspace(-0.5*Dpe, 0.5*Dpe, Nsamples)
    phase = dm1d.phase_sawtooth(xi, spatial_period, phase_wrap)
    
    plt.figure()
    plt.clf()
    plt.plot(xi, phase, '.-', label=r'$\Phi(\xi)$')
    plt.xlabel(r'$\xi [m]$')
    plt.ylabel('$n \pi$')
    plt.title('$c_2 = %g m rms$'%c2+' '+'$\Lambda = %g m$'%spatial_period + '$\Phi_0 = %g \pi$'%M)
    
    I, x = dm1d.get_diffraction_pattern(phase)
    
    ccd_pixel_pitch = 4.65e-6
    ccd_half_size = 1360 * 0.5 * ccd_pixel_pitch
    x_tilt = dm1d.get_tilted_psf_displacement(c2)
    plt.figure()
    plt.clf()
    plt.plot(x, I/I.max(),'ko-', label='fft')
    plt.vlines(x_tilt, 0, 1, colors = 'red', linestyles='dashed', label = '$f 4c_2/D_{pe}$')
    plt.ylabel('Normalized Intentisy')
    plt.xlabel('$x [m]$')
    plt.xlim(- ccd_half_size, ccd_half_size)
    plt.legend( loc = 'best')

def applied_tilt_on_slm():
    
    fname_masters = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230705\\230705cma_masters_texp_1p5ms.fits"
    cam, mirror  = sharp_psf_on_camera.create_devices()
    spoc = sharp_psf_on_camera.SharpPsfOnCamera(cam, mirror, fname_masters)
    dm1d = DiffractionModel1D()
    # pupil mask coord
    R = 571 # 5.25 mm
    yc, xc = 571, 875
    spoc.change_circular_mask((yc, xc), R)
    
    #tilt
    c2 = 500e-9
    spoc.write_zernike_on_slm([c2])
    wf_in_m = my_tools.reshape_vector2map(mirror.get_shape())
    tilt_opd_1d = wf_in_m[yc,:]
    tilt_uint8 = my_tools.convert2uint8(tilt_opd_1d, 635e-9)
    wrapped_phase = my_tools.convert_uint8_2wrapped_phase(tilt_uint8)
    
    plt.figure()
    plt.clf()
    plt.plot(wrapped_phase)
    plt.ylabel('$\Phi [rad]$')
    
    phase = wrapped_phase[wrapped_phase.mask == False]
    I, x = dm1d.get_diffraction_pattern(phase)
    
    Itot = I.sum()
    II = I/Itot
    plt.figure()
    plt.clf()
    plt.plot(x, II, 'ko-')
    plt.ylabel('Normalized Intensity')
    plt.xlabel('x [m]')
    ccd_size = 0.5*1360*4.65e-6
    plt.xlim(- ccd_size, ccd_size)
    plt.vlines( dm1d.get_tilted_psf_displacement(c2), 0, II.max(),colors='red',label='f*4c2/Dpe')
    plt.legend(loc='best')
    
    
    
    
    