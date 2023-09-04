import numpy as np


class DiffractionModel1D():
    
    def __init__(self,
                 wl_in_m = 635e-9,
                 incident_angle_in_deg = 0,
                 Dpe_in_m = 10.5e-3,
                 focal_length_in_m = 250e-3,
                 slm_pix_pitch_in_m = 9.2e-6
                 ):
        self._wl = wl_in_m
        self._alpha = incident_angle_in_deg * np.pi / 180
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
        ladder = self._single_phase_ladder(x, Nsteps, phase_wrap)
        Nladders = int(self._Dpe/x.ptp())
        phase = np.tile(ladder, Nladders)
        return phase
    
    def get_diffraction_pattern(self, phase, Npad = 16):
        #TO DO: add incident angle effect
        Ut = 1 * np.exp(1j * phase)
        Ut_extended = np.zeros(len(phase) * Npad, dtype=complex)
        Ut_extended[0 : len(Ut)] = Ut
        
        dx = self._Dpe/len(phase)
        x = np.fft.fftfreq(len(Ut_extended), dx) * self._wl * self._focal_length
        I = np.abs(np.fft.fft(Ut_extended)/(self._wl*self._focal_length))**2
        return I, x
    
    def show_diffraction_pattern(self, phase, Npad = 16):
        
        I, x =self.get_diffraction_pattern(phase, Npad)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(x, I, 'k-o', label = 'FFT')
        plt.xlabel('x[m]')
        plt.ylabel('Intensity')
        plt.grid('--', alpha = 0.3)
        
        