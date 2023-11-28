import numpy as np 
import matplotlib.pyplot as plt
from tesi_slm.diffraction_pattern.slm_diffractive_model1D import SteppedPhaseGratingModel1D
from tesi_slm.diffraction_pattern.fft4slm_diffraction_patterns import DiffractionModel1D ,DiffractionModel2D
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
from tesi_slm import my_tools

def show_maxima_and_minima_of_stebbed_grating(N, wl = 635e-9, f  = 250e-3, phase_wrap = 2*np.pi):
    
    spg = SteppedPhaseGratingModel1D()
    z = f
    pp = 9.2e-6
    LL = N * pp 
    
    c2 = spg.get_c2_from_sawtooth(LL, phase_wrap)
    dx = spg.compute_tilt_psf_displacement(c2_rms = c2)
    
    
    #number of order from diffractiom grating eq
    Norder = LL/wl
    xmax = Norder*f
    xmin = - Norder*f
    xspan = np.linspace(-5*dx, 5*dx, 10000)
    #xspan = np.linspace(xmin, xmax, 10000)
    m = np.arange(-int(Norder), int(Norder)+1)
    qmax = m * N + 0.5 * phase_wrap / np.pi
    xmax = qmax * wl*z/LL
    
    l = np.arange(-(int(Norder)-1),int(Norder))
    l = l[l!=0]
    qmin = l + 0.5 * phase_wrap/np.pi
    xmin = qmin * wl*z/LL
    
    lprime = np.arange(-(int(Norder)-2),int(Norder)-1)
    
    qprime = lprime + 0.5 + 0.5 * phase_wrap/np.pi
    xmax2 = qprime*wl*z/LL
    
    Inorm = spg.get_relativeIsincs2(xspan, LL, N, phase_wrap)
    Inorm_max = spg.get_relativeIsincs2(xmax, LL, N, phase_wrap)
    Inorm_min = spg.get_relativeIsincs2(xmin, LL, N, phase_wrap)
    Inorm_max2 = spg.get_relativeIsincs2(xmax2, LL, N, phase_wrap)
    
    
    q = np.arange(-int(Norder),int(Norder+1))
    xq=q*wl*z/LL
    Iq=spg.get_relativeIsincs2(xq, LL, N, phase_wrap)
    
    plt.figure()
    plt.clf()
    plt.plot(xspan, Inorm,'k-', label = 'N=%g'%N)
    plt.plot(xmin, Inorm_min, 'bo', label = 'minima')
    plt.plot(xmax2, Inorm_max2, 'go', label = 'secondary maxima')
    plt.plot(xmax, Inorm_max, 'ro', label='principal maxima')
    #plt.plot(xq, Iq, 'kx',markersize=100)
    plt.vlines(dx, 0, 1, 'black', '--', r"$\Delta x_{psf}$")
    plt.grid('--', alpha = 0.3) 
    plt.xlim(-5*dx, 5*dx)
    plt.ylim(0,1)
    plt.title("c2 = %g m rms" %c2)
    plt.ylabel('Normalized Intensity')
    plt.xlabel('position [m]')    
    plt.legend(loc='best')
    
    
    
    
def check_numerical_app_error(N, wl = 635e-9, f  = 250e-3, phase_wrap = 2*np.pi):
    
    
    spg = SteppedPhaseGratingModel1D()
    z = f
    pp = 9.2e-6
    LL = N * pp
    
    c2 = spg.get_c2_from_sawtooth(LL, phase_wrap)
    dx = spg.compute_tilt_psf_displacement(c2_rms = c2)
    
    xspan = np.linspace(-2.5*dx, 2.5*dx, 500)
    
    Inorm = spg.get_relativeIsincs2(xspan, LL, N, phase_wrap)
    
    
    cost2 = (LL**2 / (wl * f))**2
    arg_sin = (0.5 * phase_wrap / np.pi - LL * xspan / (wl * f))
    sinc_sq_1 = np.sinc(arg_sin)**2
    sinc_sq_2 = np.sinc(arg_sin / N)**2
    sinc_sq_3 = (np.sinc(LL * xspan /(N * wl * f)))**2
    sin_ratio2 = (np.sinc(arg_sin) / np.sinc(arg_sin / N))**2
    
    I = cost2 * sin_ratio2 * sinc_sq_3
        
    Itot = cost2
    print("sinc_sq_1 \t sinc_sq_2 \t sinc_sq_3 \t I/Itot")
    for i in np.arange(len(xspan)): 
        print("{0}\t{1}\t{2}\t{3}".format(sinc_sq_1[i],sinc_sq_2[i],sinc_sq_3[i],I[i]/Itot))
        

def show_diff_eff_sawtooth(phase_wrap):
    
    q = np.arange(-10,11)
    eta = np.sinc(q-0.5*phase_wrap/np.pi)**2
    M = 0.5*phase_wrap/np.pi
    plt.figure()
    plt.clf()
    plt.bar(q, eta, width=0.5 ,align='center',color='r',label=r'$\phi_0/2\pi =%g $'%M)
    plt.xticks(q)
    plt.xlabel('Diffraction order q')
    plt.ylabel('Diffraction efficiency')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
def look4localmaxima_with_2dfft(N, wl = 635e-9, f  = 250e-3, phase_wrap = 2*np.pi):
    
    #building circular pupil mask 
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    
    pp = 9.2e-6
    
    LL = N * pp
    
    dm1d = DiffractionModel1D()
    
    c2 = dm1d.get_c2_from_sawtooth(LL, phase_wrap)
    
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    
    phase_pattern_2d = my_tools.convert_opd2wrapped_phase(wf2display, wl, phase_wrap)
    dm2d = DiffractionModel2D()
    I, y, x = dm2d.get_diffraction_pattern(phase_pattern_2d)
    
    dm2d.show_image_plane(I, y, x)
    dm2d.show_image_profiles(I, y, x)
    return I, y, x

def get_diffraction_pattern_of_tilt(c2_m_rms, wl = 635e-9, f  = 250e-3, phase_wrap = 2*np.pi):
    
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    
    c2 = c2_m_rms
    
    zg = ZernikeGenerator(cmask_obj)
    wf2display = np.zeros(frameshape)
    wf2display = np.ma.array(data = wf2display, mask = cmask_obj.mask(), fill_value = 0)
    Z2 = zg.getZernike(2)
    wf2display = c2 * Z2
    
    phase_pattern_2d = my_tools.convert_opd2wrapped_phase(wf2display, wl, phase_wrap)
    dm2d = DiffractionModel2D()
    I, y, x = dm2d.get_diffraction_pattern(phase_pattern_2d)
    
    dm2d.show_image_plane(I, y, x)
    dm2d.show_image_profiles(I, y, x)
    return I, y, x

def get_intensity_from_double_sawtooth(x, LL1, LL2, wl=635e-9, f=250e-3):
    
    cost = (LL1+LL2)/(wl*f)
    arg_sinc_1 = LL1*x/(wl*f) - 1
    arg_sinc_2 = LL2*x/(wl*f) - 1
    arg_cos = np.pi*x*(LL1+LL2)/(wl*f)
    
    
    term1 = (LL1 * np.sinc(arg_sinc_1))**2
    term2 = (LL2 * np.sinc(arg_sinc_2))**2
    term3 = 2 * LL1 * LL2 * np.sinc(arg_sinc_1) * np.sinc(arg_sinc_2) * np.cos(arg_cos)
    I = cost**2 * (term1 + term2 + term3)
    
    return I

def show_pattern_of_double_sawtooth(LL1, LL2, wl = 635e-9, f  = 250e-3):
    q = np.arange(-30,31)
    xq = (wl*f/(LL1+LL2))*q
    
    I = get_intensity_from_double_sawtooth(xq, LL1, LL2, wl, f)
    Itot = I.sum()
    plt.figure()
    plt.clf()
    plt.bar(q, I/Itot, width=0.5 ,align='center',color='r')
    plt.xticks(q)
    plt.xlabel('Order q')
    plt.ylabel('Normalized Intensity')
    plt.xlim(-10.5,10.5)
    plt.grid('--', alpha = 0.3)
    
    mean_LL = (LL1+LL2)*0.5
    dm1d = DiffractionModel1D()
    c2 = dm1d.get_c2_from_sawtooth(mean_LL,phase_wrap=2*np.pi)
    dx = dm1d.get_tilted_psf_displacement(c2)
    plt.figure()
    plt.clf()
    plt.bar(xq, I/Itot, width=1e-3 ,align='center',color='r')
    plt.vlines(dx, 0, 1, 'black', '--', label=r"$\Delta x_{psf}$")
    plt.xlabel('position [m]')
    plt.ylabel('Normalized Intensity')
    plt.xlim(-10.5* (wl*f/(LL1+LL2)),10.5* (wl*f/(LL1+LL2)))
    plt.grid('--', alpha = 0.3)
    
    N1 = LL1/9.2e-6
    N2 = LL2/9.2e-6
    Nmean = (N1+N2)*0.5
   
    x_span = np.linspace(xq[0], xq[-1], 10000)
    Ienv = get_intensity_from_double_sawtooth(x_span, LL1, LL2, wl, f)
    plt.figure()
    plt.clf()
    plt.plot(x_span,Ienv/Itot,'k-', label = 'envelop')
    plt.plot(xq, I/Itot, 'ro', label = '$I(x_q)$')
    plt.vlines(dx, 0, 1, 'black', '--', label=r"$\Delta x_{psf}$")
    plt.xlabel('position [m]')
    plt.ylabel('Normalized Intensity')
    plt.xlim(-10.5* (wl*f/(LL1+LL2)),10.5* (wl*f/(LL1+LL2)))
    plt.grid('--', alpha = 0.3)
    plt.legend(loc='best')
    plt.title("$N_1 = %g$"%N1+'\t'+"$N_2=%g$"%N2+'\t\t'+"$<N>=%g$"%Nmean+'\t\t'+"$c_2$=%g m rms"%c2)
    
    
    
    