import numpy as np 
import matplotlib.pyplot as plt
from tesi_slm.diffraction_pattern.slm_diffractive_model1D import SteppedPhaseGratingModel1D

def show_maxima_and_minima_of_stebbed_grating(N, wl = 635e-9, f  = 250e-3, phase_wrap = 2*np.pi):
    
    spg = SteppedPhaseGratingModel1D()
    z = f
    pp = 9.2e-6
    LL = N * pp 
    
    c2 = spg.get_c2_from_sawtooth(LL, phase_wrap)
    dx = spg.compute_tilt_psf_displacement(c2_rms = c2)
    
    max_order = 2 * LL / wl 
    
    xspan = np.linspace(-5*dx, 5*dx, 1000)
    
    m = np.arange(-int(N), int(N)+1)
    qmax = m * N + 0.5 * phase_wrap / np.pi
    xmax = qmax * wl*z/LL
    
    l = np.arange(-(int(N)-1),int(N))
    l = l[l!=0]
    qmin = l + 0.5 * phase_wrap/np.pi
    xmin = qmin * wl*z/LL
    
    lprime = np.arange(-(int(N)-2),int(N)-1)
    
    qprime = lprime + 0.5 + 0.5 * phase_wrap/np.pi
    xmax2 = qprime*wl*z/LL
    
    Inorm = spg.get_relativeIsincs2(xspan, LL, N, phase_wrap)
    Inorm_max = spg.get_relativeIsincs2(xmax, LL, N, phase_wrap)
    Inorm_min = spg.get_relativeIsincs2(xmin, LL, N, phase_wrap)
    Inorm_max2 = spg.get_relativeIsincs2(xmax2, LL, N, phase_wrap)
    
    plt.figure()
    plt.clf()
    plt.plot(xspan, Inorm,'k-')
    #plt.plot(xmin, Inorm_min, 'bo', label = 'minima')
    #plt.plot(xmax2, Inorm_max2, 'go', label = 'secondary maxima')
    #plt.plot(xmax, Inorm_max, 'ro', label='principal maxima')
    plt.grid('--', alpha = 0.3)
    plt.xlim(-5*dx, 5*dx)
    #plt.ylim(0,1)
    plt.title(c2)
    plt.ylabel('Normalized Intensity')
    plt.xlabel('position [m]')    
    #plt.legend(loc='best')
    
    
    
    
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
    
    