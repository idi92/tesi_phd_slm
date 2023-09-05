import numpy as np


class DiffractionModel1D():
    
    def __init__(self,
                 wl_in_m = 635e-9,
                 Dpe_in_m = 10.5e-3,
                 focal_length_in_m = 250e-3,
                 slm_pix_pitch_in_m = 9.2e-6
                 ):
        self._wl = wl_in_m
        self._focal_length = focal_length_in_m
        self._Dpe = Dpe_in_m
        self._slm_pix_pitch = slm_pix_pitch_in_m
        
    def get_sawtooth_spatial_period_from_c2(self, c2_m_rms, phase_wrap):

        spatial_period = self.phase2opd(
            phase_wrap) * 0.25 * self._Dpe / c2_m_rms

        return spatial_period

    def get_c2_from_sawtooth(self, spatial_period_in_m, phase_wrap):

        c2_m_rms = self.phase2opd(phase_wrap) * \
            self._Dpe * 0.25 / spatial_period_in_m

        return c2_m_rms
    
    def opd2phase(self, opd):

        return 2 * np.pi * opd / self._wl

    def phase2opd(self, phase):

        return 0.5 * self._wl * phase / np.pi

    def get_tilted_psf_displacement(self, c2_rms):

        gamma = 4 * c2_rms / self._Dpe
        return self._focal_length * gamma

    def get_c2_rms_from_displacement(self, delta_x_in_m):

        return 0.25 * self._Dpe * delta_x_in_m / self._focal_length
    
    def phase_sawtooth(self, x, spatial_period, phase_wrap):

        phase = phase_wrap * x / spatial_period

        return phase % phase_wrap
    
    def _single_phase_ladder(self, x, Nsteps, phase_wrap):
        
        def rect(x):
            return np.where(abs(x) <= 0.5, 1, 0)

        step_width = x.ptp() / Nsteps
        res = np.zeros(x.shape)
        for step in np.arange(Nsteps):
            res += phase_wrap / Nsteps * step * \
                rect((x - (step + 0.5) * step_width) / step_width)
        return res
    
    def phase_ladders(self, x, Nsteps, phase_wrap):
        '''
        x is the span of the single phase ladder
        Nsteps is the number of steps in a single ladder
        phase_wrap is the phase value in which the phase wrap occours 
        '''
        ladder = self._single_phase_ladder(x, Nsteps, phase_wrap)
        Nladders = int(self._Dpe/x.ptp())
        phase = np.tile(ladder, Nladders)
        return phase
    
    def get_diffraction_pattern(self, phase, Npad = 16):
        '''
        computes the diffraction pattern from the 1D phase with fft and goodman formalism
        phase is the phase patter along an axis of the pupil plane (unmusked array!)
        
        '''
        # non ha molto senso tener conto dell angolo di incidenza nella
        # fft, l effetto consiste solo in una traslazione del pattern di 
        # diffrazione che otterei nel caso di incidenza normale
        # basta supporre che la rifessione speculare/ ordine zero caschi nel
        # centro della ccd
        # alla fine sto misurando distanze rispetto alla riflessione speculare 
        # alpha = incident_angle * np.pi / 180
        Ui = 1 
        t = 1 * np.exp(1j * phase)
        Ut = t * Ui
        Ut_extended = np.zeros(len(phase) * Npad, dtype=complex)
        Ut_extended[0 : len(Ut)] = Ut
        
        dx = self._Dpe/len(phase)
        x = np.fft.fftfreq(len(Ut_extended), dx) * self._wl * self._focal_length
        I = np.abs(np.fft.fft(Ut_extended)/(self._wl*self._focal_length))**2
        return I, x
    
    def show_diffraction_pattern(self, phase, Npad = 16):
        
        I, x = self.get_diffraction_pattern(phase, Npad)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(x, I, 'k-o', label = 'FFT')
        plt.xlabel('x[m]')
        plt.ylabel('Intensity')
        plt.grid('--', alpha = 0.3)
        
class DiffractionModel2D():
    
    def __init__(self,
                 wl_in_m = 635e-9,
                 Dpe_in_m = 10.5e-3,
                 focal_length_in_m = 250e-3,
                 slm_pix_pitch_in_m = 9.2e-6
                 ):
        self._wl = wl_in_m
        self._focal_length = focal_length_in_m
        self._Dpe = Dpe_in_m
        self._slm_pix_pitch = slm_pix_pitch_in_m
        
    def get_diffraction_pattern(self, phase, Npad = 16):
        
        Ui = 1 
        t = 1 * np.exp(1j * phase)
        Ut = t * Ui
        Ut.data[Ut.mask == True] = 0
        Dim = np.max(phase.shape)
        Ut_extended = np.zeros((Dim * Npad, Dim * Npad), dtype=complex)
        Ut_extended[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut
        
        dxi = self._slm_pix_pitch #self._Dpe/(2*571)
        deta = dxi #self._Dpe/(2*571)
        
        x = np.fft.fftshift(np.fft.fftfreq(Ut_extended.shape[1], dxi)) * self._wl * self._focal_length
        y = np.fft.fftshift(np.fft.fftfreq(Ut_extended.shape[0], deta)) * self._wl * self._focal_length
        
        I = np.abs(np.fft.fftshift(np.fft.fft2(Ut_extended))/(self._wl*self._focal_length))**2
        
        return I, y, x
    
    def show_imageplane(self, I, y, x, camshape =(1024,1360), pixel_pitch = 4.65e-6):
        
        dy = np.abs(y[-1]- y[-2])
        dx = np.abs(x[-1]- x[-2])
        
        
        extent_pixel =[-0.5*I.shape[0]*dy/pixel_pitch, \
                  0.5*I.shape[0]*dy/pixel_pitch, \
                  -0.5*I.shape[1]*dx/pixel_pitch, \
                  0.5*I.shape[1]*dx/pixel_pitch]
        
        #extent_meters = extent_pixel * pixel_pitch
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.imshow(I, cmap = 'jet', extent=extent_pixel)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('W/m2')
        plt.xlim(-0.5*camshape[1], 0.5*camshape[1])
        plt.ylim(-0.5*camshape[0], 0.5*camshape[0])
        
        
            