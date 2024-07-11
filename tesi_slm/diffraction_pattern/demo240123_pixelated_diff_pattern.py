import numpy as np
import matplotlib.pyplot as plt


class TiltDiffractionPattern():
    
    def __init__(self, wl = 635e-9, c2_rms_m = 1e-6, a = 9e-6, d = 9.2e-6, Dpe=10.5e-3, z = 250e-3):
        self._wl = wl
        self._c2 = c2_rms_m
        self._a = a
        self._d = d
        self._Dpe = Dpe
        self._z = z
        self._N = int(np.round(Dpe/d))
        
    def intensity_due_tilt(self, x, phi = 2*np.pi):
        
        self._phi = phi
        #self._Dpe = 1152*self._d
        LL0 = self._wl*self._Dpe/(4*self._c2)
        LL = LL0*self._phi/(2*np.pi)
        self._LL = LL
        
        fx = (x/(self._wl*self._z))
        ratio = self._phi/(2*np.pi*LL)
        
        cost2 = (self._a*self._N/(self._wl*self._z))**2
        
        arg_a = self._d*(fx- ratio)
        sinc1 = np.sinc(arg_a)**2
        sinc2 = np.sinc(self._N*arg_a)**2
        sinc3 = np.sinc(self._a*(fx- ratio))**2
        
        I = cost2*sinc2/sinc1*sinc3
        return I
        
    def show_diffraction_pattern(self, x, I):
        m = self._phi/np.pi*0.5
        l = np.arange(1,10)
        xrel = self._wl*self._z*(((2*l+1)/(2*self._N*self._d)) + (self._phi*0.5/(np.pi*self._LL)))
        plt.figure()
        plt.clf()
        plt.plot(x,I,'ko', label=r"$\phi/2\pi = %g$"%m)
        
        x_tilt = 4*self._c2/self._Dpe*self._z
        plt.vlines(x_tilt, I.min(), I.max(), colors='red',linestyles = 'dashed',label=r"$\Delta x_{tilt}$")
        plt.vlines(xrel, I.min(), I.max(), colors='blue',linestyles = 'dashed',label=r"$\x_{rel}$")     
        plt.grid()
        plt.xlabel('position [m]')
        plt.ylabel('Intensity')
        plt.legend(loc='best')
        plt.title("tilt c2 = %g m rms"%self._c2)
        
        