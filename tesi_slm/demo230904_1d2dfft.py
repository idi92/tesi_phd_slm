import numpy as np
from tesi_slm.fft4slm_diffraction_patterns import DiffractionModel1D, DiffractionModel2D
from tesi_slm import sharp_psf_on_camera, my_tools, slm_diffractive_model1D
import matplotlib.pyplot as plt
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
from tesi_slm import my_tools

def tilt_fft():
    
    phase_wrap = 2 * np.pi
    M = phase_wrap / 2* np.pi
    dm1d = DiffractionModel1D()
    c2 = 500e-9
    spatial_period = dm1d.get_sawtooth_spatial_period_from_c2(c2, phase_wrap)
    
    Dpe = 10.5e-3
   
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
    

def considering_incident_angle():
    # non ha molto senso tener conto dell angolo di incidenza nella
    # fft, l effetto consiste solo in una traslazione del pattern di 
    # diffrazione che otterei nel caso di incidenza normale
    # basta supporre che la rifessione speculare/ ordine zero caschi nel
    # centro della ccd
    # alla fine sto misurando distanze rispetto alla riflessione speculare 
    
    #building circular pupil mask 
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    c2 = 500e-9
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    # considering 1D case and converting to wrapped phase on slm
    wf1d = wf2display[571,:]
    wf1d_uint8 = my_tools.convert2uint8(wf1d, 635e-9)
    wrapped_phase1d = my_tools.convert_uint8_2wrapped_phase(wf1d_uint8, 2*np.pi)
   
    dm1d = DiffractionModel1D()
    phase1d = wrapped_phase1d[wrapped_phase1d.mask == False]
    
    I, x = dm1d.get_diffraction_pattern(phase1d)
    
    #alpha = alpha_deg/180*np.pi
    plt.figure()
    plt.clf()
    plt.plot(x, I, 'ko-')
    plt.ylabel('Intensity')
    plt.xlabel('x [m]')
    ccd_size = 0.5*1360*4.65e-6
    plt.xlim(- ccd_size, ccd_size)
    plt.vlines( dm1d.get_tilted_psf_displacement(c2), 0, I.max(),colors='red',label='f*4c2/Dpe')
    plt.legend(loc='best')
    plt.title('$c_2 = %g m rms$'%c2)
    
    slm_halfsize = 0.5 * 1920 * 9.2e-6
    xi = np.linspace(-slm_halfsize, slm_halfsize, len(wrapped_phase1d))
    xi = np.ma.array(data = xi, mask = wrapped_phase1d.mask)
    plt.figure()
    plt.clf()
    plt.plot(xi, wrapped_phase1d,'.-')
    plt.xlabel(r'$xi[m]$')
    plt.ylabel(r'$\Phi(\xi)[rad]$')
    plt.title('$c_2 = %g m rms$'%c2)
    
def fft2d(c2 = 4.16e-6, phase_wrap = 2*np.pi):
    #building circular pupil mask 
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    #c2 = 4.16e-6
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    
    #wfuint8 = my_tools.convert2uint8(wf2display, 635e-9)
    wrapped_phase = my_tools.convert_opd2wrapped_phase(wf2display, 635e-9, phase_wrap)
    
    dm2d = DiffractionModel2D()
    
    I, y, x = dm2d.get_diffraction_pattern(wrapped_phase, 4)
    dm2d.show_image_plane(I, y, x)

def zero_first_order1Dmodel_I_vs_phi(c2, phase_span):
    
    spgm1d = slm_diffractive_model1D.SteppedPhaseGratingModel1D()
    I0 = np.zeros(len(phase_span))
    I1 = np.zeros(len(phase_span))
    for idx, phi in enumerate(phase_span):
        
        LL = spgm1d.get_sawtooth_spatial_period_from_c2(c2, phi)
        N = min(LL/spgm1d._slm_pix_pitch, 256)
        x0 = spgm1d.get_sampling_points_from_comb(LL, 0)
        x1 = spgm1d.get_sampling_points_from_comb(LL, 1)
        #I0[idx] = spgm1d.sincs2_intensity_pattern1D(x0, LL, N, phi)
        #I1[idx] = spgm1d.sincs2_intensity_pattern1D(x1, LL, N, phi)
        I0[idx] = spgm1d.get_relativeIsincs2(x0, LL, N, phi)
        I1[idx] = spgm1d.get_relativeIsincs2(x1, LL, N, phi)
    return I0, I1
    
def zero_first_ordes_as_a_func_of_phi_wrap(phase_span, c2 = 4.16e-6):
    #building circular pupil mask 
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    #c2 = 4.16e-6
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    
    dm2d = DiffractionModel2D()
    I_at_zero = np.zeros(len(phase_span))
    I_at_first =  np.zeros(len(phase_span))
    
    for idx, phi in enumerate(phase_span):
        
        wrapped_phase = my_tools.convert_opd2wrapped_phase(wf2display, 635e-9, phi)
        
        I, y, x = dm2d.get_diffraction_pattern(wrapped_phase, 4)
        Idl_max = dm2d._Idl.max()
        yd, xd = my_tools.get_index_from_image(dm2d._Idl)
        y1, x1 = my_tools.get_index_from_image(I)
        I_at_zero[idx] = I[yd,xd]/Idl_max
        I_at_first[idx] = I[y1,x1]/Idl_max
        
    return I_at_zero, I_at_first

def plot_zero_and_first_order_vs_phi():
    phase_span = 2*np.pi * np.linspace(0.5,1,10)
    I0, I1 = zero_first_ordes_as_a_func_of_phi_wrap(phase_span, c2 = 4.16e-6)
    plt.figure()
    plt.clf()
    plt.plot(phase_span, I0,'bo-',phase_span,I1,'ro-')
    
def show_expected_order_efficiency(phase_wrap=2*np.pi):
    camshape =(1024,1360)
    pixel_pitch = 4.65e-6
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    N=13
    c2 = np.linspace(-15e-6,15e-6,N)
    Z2 = zg.getZernike(2)
    dm2d = DiffractionModel2D()
    
    plt.figure()
    plt.clf()
    xdomain = 0.5 * camshape[1] * pixel_pitch
    ydomain = 0.5 * camshape[0] * pixel_pitch
    
    eta1 = np.zeros(N)
    eta0 = np.zeros(N)    
    
    for idx, amp in enumerate(c2):
        wf2display = amp * Z2
        wrapped_phase = my_tools.convert_opd2wrapped_phase(wf2display, 635e-9, phase_wrap)
        I, y, x = dm2d.get_diffraction_pattern(wrapped_phase, 8)
        Idl_max = dm2d._Idl.max()
        yc, xc = np.where(dm2d._Idl == Idl_max)[0][0], np.where(dm2d._Idl==Idl_max)[1][0]
        plt.plot(x, I[yc,:]/Idl_max, '.-', label='%g um rms'%amp)
        
        eta1[idx] = I.max()/Idl_max
        eta0[idx] = I[yc,xc]/Idl_max
        
    plt.xlim(-xdomain, xdomain)
    plt.xlabel('x [m]')
    plt.ylabel('Normalized Intensity')
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(c2, eta1, 'ro-', label=r'$\eta_1$')
    plt.plot(c2,eta0,'bo-',label=r'$\eta_0$')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    plt.xlabel('$c_2[um]rms$')
    plt.ylabel(r'$\eta$') 
    
def _get_phase_pattern_c2(c2=4.16e-6, phase_wrap=2*np.pi):
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    wrapped_phase = my_tools.convert_opd2wrapped_phase(wf2display, 635e-9, phase_wrap)
    return wrapped_phase

def simulating_pixelated_slm_structure1d(c2_m_rms):
    camshape =(1024,1360)
    d = 9.2e-6
    delta = 9e-6
    f = 250e-3
    wl = 635e-9
    c2 = c2_m_rms
    Dpe = 10.5e-3
    LL = Dpe/(4*c2)*wl
    
    xdomain = 0.5 * camshape[1] * 4.65e-6
    xx = np.linspace(-0.1,0.1,5000)
    def isinc2(x):
        d = 9.2e-6
        delta = 9e-6
        f = 250e-3
        wl = 635e-9
        c2 = 15e-6
        Dpe = 10.5e-3
        LL = Dpe/(4*c2)*wl
        arg = delta*(x/(f*wl) - 1/LL)
        I = (delta*d/(f*wl))**2*np.sinc(arg)**2
        return I

    qmax = int(xdomain/(wl*f/d))
    xq = np.arange(-2,2+1)*wl*f/d
    
    plt.figure()
    plt.clf()
    plt.plot(xx, isinc2(xx), 'k.-')
    plt.plot(xq, isinc2(xq), 'ro')
    plt.xlabel('x axis [m]')
    plt.ylabel('Intensity') 
    