import numpy as np


class SawtoothDiffractiveModel1D():

    def __init__(self,
                 wl_in_m=635e-9,
                 incident_angle_in_deg=0,
                 Dpe_in_m=10.53e-3,
                 focal_length_in_m=250e-3,
                 slm_pix_pitch_in_m=9.2e-6
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

    def phase_sawtooth1D(self, x, spatial_period, phi_max):

        phase = phi_max * x / spatial_period

        return phase % phi_max

    def sinc2_intensity1D(self, x, spatial_period, phi_max):

        sinc_arg = (spatial_period * x / (self._wl *
                                          self._focal_length)) - 0.5 * phi_max / np.pi
        sinc2 = np.sinc(sinc_arg)**2
        cost2 = (spatial_period**2 / (self._wl * self._focal_length))**2

        return cost2 * sinc2

    def get_sampling_points_from_comb(self, spatial_period, q):

        return q * self._wl * self._focal_length / spatial_period

    def get_linspace_image_plane_axis(self, Npoints, cam_shape=(1024, 1360), pixel_pitch_in_m=4.65e-6):

        ymax = cam_shape[0] * 0.5 * pixel_pitch_in_m
        ymin = - ymax
        xmax = cam_shape[1] * 0.5 * pixel_pitch_in_m
        xmin = - xmax
        y_span = np.linspace(ymin, ymax, Npoints)
        x_span = np.linspace(xmin, xmax, Npoints)

        return y_span, x_span

    def get_visible_orders_on_ccd(self, spatial_period, NpixelOn1Axis=1360, pixel_pitch_in_m=4.65e-6):

        xmax = NpixelOn1Axis * 0.5 * pixel_pitch_in_m
        qmax = int(xmax * spatial_period / (self._wl * self._focal_length))
        Norders = qmax * 2 + 1
        q = np.linspace(-qmax, qmax, Norders)

        return q

    def show_phase_pattern(self, x, spatial_period, phi_max):

        M = phi_max / np.pi
        c2_rms = self.get_c2_from_sawtooth(spatial_period, phi_max)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.clf()
        phase = self.phase_sawtooth1D(x, spatial_period, phi_max)
        plt.plot(x, phase, '-')
        plt.grid('--', alpha=0.3)
        plt.title('Phase pattern:' + '\t' + '$\Phi_{max}= %g \pi$' % M + '\t' +
                  '$\Lambda = %g m$' % spatial_period + '\t' + '$c_2 = %g m$' % c2_rms)
        plt.xlabel(r'$\xi$' + ' ' + '[m]')
        plt.ylabel(r'$\Phi(\xi)$' + ' ' + '[rad]')

    def show_intensity_pattern(self, x, spatial_period, phi_max, NpixelOn1Axis=1360, pixel_pitch_in_m=4.65e-6):

        M = phi_max / np.pi
        I = self.sinc2_intensity1D(x, spatial_period, phi_max)

        q_orders = self.get_visible_orders_on_ccd(
            spatial_period, NpixelOn1Axis, pixel_pitch_in_m)
        xq = self.get_sampling_points_from_comb(spatial_period, q_orders)
        Iq = self.sinc2_intensity1D(xq, spatial_period, phi_max)
        Itot = Iq.sum()
        eta_q = Iq / Itot

        c2_rms = self.get_c2_from_sawtooth(spatial_period, phi_max)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(x, I, '-k', label='sinc2')
        plt.plot(xq, Iq, 'go', label='comb')
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        plt.title('Diffraction pattern:' + '\t' + '$\Phi_{max}= %g \pi$' % M + '\t' +
                  '$\Lambda = %g m$' % spatial_period + '\t' + '$c_2 = %g m$' % c2_rms)
        plt.xlabel(r'$x$' + ' ' + '[m]')
        plt.ylabel(r'$Intensity$')

        plt.figure()
        plt.clf()
        plt.bar(q_orders, eta_q, align='center', color='g')
        plt.title('Diffraction efficiency:' + '\t' + '$\Phi_{max}= %g \pi$' % M +
                  '\t' + '$\Lambda = %g m$' % spatial_period + '\t' + '$c_2 = %g m$' % c2_rms)
        plt.grid('--', alpha=0.3)
        plt.xlabel(r'$q$' + ' ' + 'order')
        plt.ylabel(r'$\eta_{q} = I_q/\Sigma I_{q}$')
        plt.xticks(q_orders)

    def opd2phase(self, opd):

        return 2 * np.pi * opd / self._wl

    def phase2opd(self, phase):

        return 0.5 * self._wl * phase / np.pi

    def compute_tilt_psf_displacement(self, c2_rms):

        gamma = 4 * c2_rms / self._Dpe
        return self._focal_length * gamma


class SteppedPhaseGratingModel1D():

    def __init__(self,
                 wl_in_m=635e-9,
                 incident_angle_in_deg=0,
                 Dpe_in_m=10.5e-3,
                 focal_length_in_m=250e-3,
                 slm_pix_pitch_in_m=9.2e-6
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

    def _TRASHME_stapped_phase_grating(self, x, Nsteps, spatial_period, phase_wrap):

        n = np.linspace(0, Nsteps - 1, Nsteps)
        phase = x * phase_wrap / Nsteps

        return 0

    def sincs2_intensity_pattern1D(self, x, spatial_period, Nsteps, phase_wrap):

        cost2 = (spatial_period**2 / (self._wl * self._focal_length))**2
        arg_sin = (0.5 * phase_wrap / np.pi - spatial_period *
                   x / (self._wl * self._focal_length))
        sin_ratio2 = (np.sinc(arg_sin) / np.sinc(arg_sin / Nsteps))**2
        sinc2 = (np.sinc(spatial_period * x /
                         (Nsteps * self._wl * self._focal_length)))**2
        I = cost2 * sin_ratio2 * sinc2

        return I
    
    def get_relativeIsincs2(self, x, spatial_period, Nsteps, phase_wrap):
        
        Itot = (spatial_period**2 / (self._wl * self._focal_length))**2
        I = self.sincs2_intensity_pattern1D(x, spatial_period, Nsteps, phase_wrap)
        return I/Itot
    
    def get_sampling_points_from_comb(self, spatial_period, q):

        return q * self._wl * self._focal_length / spatial_period

    def get_linspace_image_plane_axis(self, Npoints, cam_shape=(1024, 1360), pixel_pitch_in_m=4.65e-6):

        ymax = cam_shape[0] * 0.5 * pixel_pitch_in_m
        ymin = - ymax
        xmax = cam_shape[1] * 0.5 * pixel_pitch_in_m
        xmin = - xmax
        y_span = np.linspace(ymin, ymax, Npoints)
        x_span = np.linspace(xmin, xmax, Npoints)

        return y_span, x_span

    def get_visible_orders_on_ccd(self, spatial_period, NpixelOn1Axis=1360, pixel_pitch_in_m=4.65e-6):

        xmax = NpixelOn1Axis * 0.5 * pixel_pitch_in_m
        qmax = int(xmax * spatial_period / (self._wl * self._focal_length))
        Norders = qmax * 2 + 1
        q = np.linspace(-qmax, qmax, Norders)

        return q

    def opd2phase(self, opd):

        return 2 * np.pi * opd / self._wl

    def phase2opd(self, phase):

        return 0.5 * self._wl * phase / np.pi

    def compute_tilt_psf_displacement(self, c2_rms):

        gamma = 4 * c2_rms / self._Dpe
        return self._focal_length * gamma

    def get_c2_rms_from_displacement(self, delta_x_in_m):

        return 0.25 * self._Dpe * delta_x_in_m / self._focal_length

    def get_diffraction_efficiency(self, q_order, Nsteps, spatial_period, phase_wrap, NpixelOn1Axis=1360, pixel_pitch_in_m=4.65e-6):

        orders = self.get_visible_orders_on_ccd(
            spatial_period, NpixelOn1Axis, pixel_pitch_in_m)
        x = self.get_sampling_points_from_comb(spatial_period, orders)
        I = self.sincs2_intensity_pattern1D(
            x, spatial_period, Nsteps, phase_wrap)
        Itot = I.sum()
        eta = I / Itot
        q = np.where(q_order == orders)[0][0]

        return eta[q]

    def show_intensity_pattern(self, x, Nsteps, spatial_period, phase_wrap, NpixelOn1Axis=1360, pixel_pitch_in_m=4.65e-6):

        M = phase_wrap / np.pi
        I = self.sincs2_intensity_pattern1D(
            x, spatial_period, Nsteps, phase_wrap)

        q_orders = self.get_visible_orders_on_ccd(
            spatial_period, NpixelOn1Axis, pixel_pitch_in_m)
        xq = self.get_sampling_points_from_comb(spatial_period, q_orders)
        Iq = self.sincs2_intensity_pattern1D(
            xq, spatial_period, Nsteps, phase_wrap)
        Itot = Iq.sum()
        eta_q = Iq / Itot

        c2_rms = self.get_c2_from_sawtooth(spatial_period, phase_wrap)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.semilogy(x, I, '-k', label='sinc2sinc2/sinc2')
        plt.semilogy(xq, Iq, 'ro', label='comb')
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        plt.title('Diffraction pattern:' + '\t' + '$\Phi_{max}= %g \pi$' % M + '\t' +
                  '$\Lambda = %g m$' % spatial_period + '\t' + '$c_2 = %g m$' % c2_rms)
        plt.xlabel(r'$x$' + ' ' + '[m]')
        plt.ylabel(r'$Intensity$')

        plt.figure()
        plt.clf()
        plt.bar(q_orders, eta_q, align='center', color='r')
        plt.title('Diffraction efficiency:' + '\t' + '$\Phi_{max}= %g \pi$' % M +
                  '\t' + '$\Lambda = %g m$' % spatial_period + '\t' + '$c_2 = %g m$' % c2_rms)
        plt.grid('--', alpha=0.3)
        plt.xlabel(r'$q$' + ' ' + 'order')
        plt.ylabel(r'$\eta_{q} = I_q/\Sigma I_{q}$')
        plt.xticks(q_orders)


def exampleWithFFT():
    wv = 633e-9

    # opd spatial profile in a 10mm pupil sampled at 1um
    xx = np.linspace(0, 2e-3, 2000)
    opd = np.tile(0.9 * wv / 2e-3 * xx, 5)

    # phase and electric field
    phase = (opd / wv % 1) * 2 * np.pi
    ef = 1 * np.exp(1j * phase)

    # pad 16 times to increase spatial resolution in image plane
    efext = np.zeros(len(phase) * 16, dtype=complex)
    efext[0:len(phase)] = ef

    # spatial sampling in the pupil
    x = np.linspace(-5e-3, 4.999e-3, len(phase))

    import matplotlib.pyplot as plt

    # plot image intensity computed as FT of field in the pupil
    plt.plot(np.fft.fftfreq(len(efext), 1e-6) * wv *
             250e-3, np.abs(np.fft.fft(efext))**2, '.-')
    plt.grid(True)
    plt.xlim(-5e-3, 5e-3)


def opd_ladder(x, n_steps, amplitude):

    def rect(x):
        return np.where(abs(x) <= 0.5, 1, 0)

    step_width = x.ptp() / n_steps
    res = np.zeros(x.shape)
    for step in np.arange(n_steps):
        res += amplitude / n_steps * step * \
            rect((x - (step + 0.5) * step_width) / step_width)
    return res


def example2WithFFT():
    wv = 633e-9
    xx = np.linspace(0, 2.5e-3, 2500)
    opd = np.tile(opd_ladder(xx, 256, 1.0 * wv), 4)
    phase = (opd / wv % 1) * 2 * np.pi
    ef = 1 * np.exp(1j * phase)
    efext = np.zeros(len(phase) * 16, dtype=complex)
    efext[0:len(phase)] = ef
    x = np.linspace(-5e-3, 4.999e-3, len(phase))

    import matplotlib.pyplot as plt
    plt.plot(np.fft.fftfreq(len(efext), 1e-6) * wv *
             250e-3, np.abs(np.fft.fft(efext))**2, '.-')
    plt.grid(True)
    plt.xlim(-2e-3, 4e-3)


def example3WithFFT():
    c2 = 10e-6
    wv = 633e-9
    x = np.linspace(-5e-3, 4.99e-3, 1000)
    opd = 4 * c2 / 10e-3 * x
    opd_d = (np.round(opd / wv * 256) % 256)
    phase = opd_d / 256 * 2 * np.pi
    ef = 1 * np.exp(1j * phase)
    efext = np.zeros(len(x) * 16, dtype=complex)
    efext[0:len(x)] = ef

    import matplotlib.pyplot as plt
    plt.plot(np.fft.fftfreq(len(efext), 10e-6) * wv *
             250e-3, np.abs(np.fft.fft(efext))**2, '.-')
    plt.grid(True)
    plt.xlim(-2e-3, 4e-3)
