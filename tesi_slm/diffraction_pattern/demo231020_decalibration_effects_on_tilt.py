import numpy as np
import matplotlib.pyplot as plt
from tesi_slm.diffraction_pattern import demo230904_1d2dfft
from scipy.interpolate import CubicSpline
from astropy.io import fits

def main(c2 = 4.16e-6, fname='pippo.fits'):
    phase_span = 2*np.pi*np.linspace(0.5, 1, 50)
    #relative intensities
    #FFT2D
    I0, I1 = demo230904_1d2dfft.zero_first_ordes_as_a_func_of_phi_wrap(phase_span)
    #1d ANALYTICAL SOLUTION
    I0_1d, I1_1d =demo230904_1d2dfft.zero_first_order1Dmodel_I_vs_phi(4.16e-6,phase_span)
    
    plt.figure()
    plt.clf()
    plt.plot(phase_span, I0,'bx')
    plt.plot(phase_span, I1,'rx')
    plt.plot(phase_span, I0_1d,'b-', label='$\eta_0$')
    plt.plot(phase_span, I1_1d,'r-', label='$\eta_1$')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    plt.ylabel('$\eta_q=I_q/I_{tot}$')
    plt.xlabel('$\phi_0 [rad]$')
    
    hdr = fits.Header()
    hdr['C2_M_RMS'] = c2
    fits.writeto(fname , I0, hdr)
    fits.append(fname, I1)
    fits.append(fname, I0_1d)
    fits.append(fname, I1_1d)
    fits.append(fname, phase_span)
    
    #I0int = CubicSpline(phase_span, I0_1d, bc_type='natural')
    #I1int = CubicSpline(phase_span, I1_1d, bc_type='natural')
    
def plot_eta_vs_phi(phase_span, I0, I1, I0_1d, I1_1d):
    
    phase_n = 0.5*phase_span/np.pi
    plt.figure()
    plt.clf()
    plt.plot(phase_n, I0,'bx')
    plt.plot(phase_n, I1,'rx')
    plt.plot(phase_n, I0_1d,'b-', label='$\eta_0$')
    plt.plot(phase_n, I1_1d,'r-', label='$\eta_1$')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    plt.ylabel('$\eta_q = I_q/I_{tot}$')
    plt.xlabel('$\phi_0 / 2\pi$')
