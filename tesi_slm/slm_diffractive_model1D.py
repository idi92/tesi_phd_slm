import numpy as np

class SawtoothDiffractiveModel1D():
    
    def __init__(self,
                wl_in_m = 632.8e-9,
                incident_angle_in_deg = 0,
                Dpe_in_m = 10.53e-3,
                focal_length_in_m = 250e-3,
                slm_pix_pitch_in_m = 9.2e-6
                ):
        self._wl = wl_in_m
        self._alpha = incident_angle_in_deg * np.pi/180
        self._focal_length = focal_length_in_m
        self._Dpe = Dpe_in_m
        self._slm_pix_pitch = slm_pix_pitch_in_m
        

    def get_sawtooth_spatial_period_from_c2(self, c2_m_rms, phase_wrap):
        
        spatial_period = self.phase2opd(phase_wrap) * 0.25* self._Dpe /c2_m_rms 
        
        return spatial_period
    
    def get_c2_from_sawtooth(self, spatial_period_in_m, phase_wrap):
        
        c2_m_rms = self.phase2opd(phase_wrap) * self._Dpe * 0.25 / spatial_period_in_m 
        
        return c2_m_rms
        
    def phase_sawtooth1D(self, x, spatial_period, phi_max):
        
        phase = phi_max * x /spatial_period
        
        return phase % phi_max 
    
    def sinc2_intensity1D(self, x, spatial_period, phi_max):
        
        sinc_arg = (spatial_period * x / (self._wl * self._focal_length)) - 0.5* phi_max / np.pi
        sinc2 = np.sinc(sinc_arg)**2
        cost2 = (spatial_period**2/(self._wl*self._focal_length))**2
        
        return cost2*sinc2
    
    def get_sampling_points_from_comb(self, spatial_period, q):
        
        return q * self._wl*self._focal_length/spatial_period
    
    def get_linspace_image_plane_axis(self, Npoints, cam_shape = (1024, 1360), pixel_pitch_in_m = 4.65e-6):
        
        ymax = cam_shape[0] * 0.5 * pixel_pitch_in_m
        ymin = - ymax
        xmax = cam_shape[1] * 0.5 * pixel_pitch_in_m
        xmin = - xmax 
        y_span = np.linspace(ymin, ymax, Npoints)
        x_span = np.linspace(xmin, xmax, Npoints)
        
        return y_span, x_span
    
    def get_visible_orders_on_ccd(self, spatial_period, NpixelOn1Axis = 1360 , pixel_pitch_in_m = 4.65e-6):
        
        xmax = NpixelOn1Axis * 0.5 * pixel_pitch_in_m
        qmax = int(xmax * spatial_period/ (self._wl * self._focal_length))
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
        plt.grid('--', alpha = 0.3)
        plt.title('Phase pattern:'+'\t'+'$\Phi_{max}= %g \pi$'%M +'\t' +'$\Lambda = %g m$'%spatial_period + '\t' + '$c_2 = %g m$'%c2_rms)
        plt.xlabel(r'$\xi$'+' '+'[m]')
        plt.ylabel(r'$\Phi(\xi)$' + ' '+'[rad]')
        
    def show_intensity_pattern(self, x, spatial_period, phi_max, NpixelOn1Axis = 1360 , pixel_pitch_in_m = 4.65e-6):
        
        M = phi_max / np.pi
        I = self.sinc2_intensity1D(x, spatial_period, phi_max)
        
        q_orders = self.get_visible_orders_on_ccd(spatial_period, NpixelOn1Axis, pixel_pitch_in_m) 
        xq = self.get_sampling_points_from_comb(spatial_period, q_orders)
        Iq = self.sinc2_intensity1D(xq, spatial_period, phi_max)
        Itot = Iq.sum()
        eta_q = Iq/Itot
        
        c2_rms = self.get_c2_from_sawtooth(spatial_period, phi_max)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(x, I, '-k', label = 'sinc2')
        plt.plot(xq, Iq, 'go', label = 'comb')
        plt.legend(loc = 'best')
        plt.grid('--', alpha = 0.3)
        plt.title('Diffraction pattern:'+'\t'+'$\Phi_{max}= %g \pi$'%M +'\t' +'$\Lambda = %g m$'%spatial_period + '\t' + '$c_2 = %g m$'%c2_rms)
        plt.xlabel(r'$x$'+' '+'[m]')
        plt.ylabel(r'$Intensity$')
        
        plt.figure()
        plt.clf()
        plt.bar(q_orders, eta_q, align = 'center', color ='g')
        plt.title('Diffraction efficiency:'+'\t'+'$\Phi_{max}= %g \pi$'%M +'\t' +'$\Lambda = %g m$'%spatial_period+ '\t' + '$c_2 = %g m$'%c2_rms)
        plt.grid('--', alpha = 0.3)
        plt.xlabel(r'$q$'+' '+'order')
        plt.ylabel(r'$\eta_{q} = I_q/\Sigma I_{q}$')
        plt.xticks(q_orders)
    
    def opd2phase(self, opd):
        
        return 2 * np.pi * opd / self._wl
    
    def phase2opd(self, phase):
        
        return 0.5* self._wl * phase / np.pi
    
    def compute_tilt_psf_displacement(self, c2_rms):
        
        gamma = 4 * c2_rms / self._Dpe
        return self._focal_length * gamma

class SteppedPhaseGratingModel1D():
       
    def __init__(self,
                wl_in_m = 632.8e-9,
                incident_angle_in_deg = 0,
                Dpe_in_m = 10.53e-3,
                focal_length_in_m = 250e-3,
                slm_pix_pitch_in_m = 9.2e-6
                ):
        self._wl = wl_in_m
        self._alpha = incident_angle_in_deg * np.pi/180
        self._focal_length = focal_length_in_m
        self._Dpe = Dpe_in_m
        self._slm_pix_pitch = slm_pix_pitch_in_m

    def get_sawtooth_spatial_period_from_c2(self, c2_m_rms, phase_wrap):
        
        spatial_period = self.phase2opd(phase_wrap) * 0.25* self._Dpe /c2_m_rms 
        
        return spatial_period
    
    def get_c2_from_sawtooth(self, spatial_period_in_m, phase_wrap):
        
        c2_m_rms = self.phase2opd(phase_wrap) * self._Dpe * 0.25 / spatial_period_in_m 
        
        return c2_m_rms

    def _stapped_phase_grating(self, x, Nsteps, spatial_period, phase_wrap):
        
        n = np.linspace(0, Nsteps-1, Nsteps)
        phase = x * phase_wrap/Nsteps
        
        return 0
    
    def sins_sinc_intensity_pattern1D(self, x, spatial_period, Nsteps, phase_wrap):
        
        cost2 = (spatial_period**2/(self._wl*self._focal_length))**2
        arg_sin = (0.5 * phase_wrap/np.pi - spatial_period* x / (self._wl * self._focal_length))
        sin_ratio2 = (np.sinc(arg_sin) / np.sinc(arg_sin / Nsteps))**2
        sinc2 = (np.sinc(spatial_period * x/(Nsteps * self._wl * self._focal_length)))**2
        I = cost2 * sin_ratio2 * sinc2
        
        return I
    
    def get_sampling_points_from_comb(self, spatial_period, q):
        
        return q * self._wl*self._focal_length/spatial_period
    
    def get_linspace_image_plane_axis(self, Npoints, cam_shape = (1024, 1360), pixel_pitch_in_m = 4.65e-6):
        
        ymax = cam_shape[0] * 0.5 * pixel_pitch_in_m
        ymin = - ymax
        xmax = cam_shape[1] * 0.5 * pixel_pitch_in_m
        xmin = - xmax 
        y_span = np.linspace(ymin, ymax, Npoints)
        x_span = np.linspace(xmin, xmax, Npoints)
        
        return y_span, x_span
    
    def get_visible_orders_on_ccd(self, spatial_period, NpixelOn1Axis = 1360 , pixel_pitch_in_m = 4.65e-6):
        
        xmax = NpixelOn1Axis * 0.5 * pixel_pitch_in_m
        qmax = int(xmax * spatial_period/ (self._wl * self._focal_length))
        Norders = qmax * 2 + 1
        q = np.linspace(-qmax, qmax, Norders)
        
        return q 
    
    def opd2phase(self, opd):
        
        return 2 * np.pi * opd / self._wl
    
    def phase2opd(self, phase):
        
        return 0.5* self._wl * phase / np.pi
    
    def compute_tilt_psf_displacement(self, c2_rms):
        
        gamma = 4 * c2_rms / self._Dpe
        return self._focal_length * gamma
    
    def show_intensity_pattern(self, x, Nsteps, spatial_period, phase_wrap, NpixelOn1Axis = 1360 , pixel_pitch_in_m = 4.65e-6):
        
        M = phase_wrap / np.pi
        I = self.sins_sinc_intensity_pattern1D(x, spatial_period, Nsteps, phase_wrap)
        
        q_orders = self.get_visible_orders_on_ccd(spatial_period, NpixelOn1Axis, pixel_pitch_in_m) 
        xq = self.get_sampling_points_from_comb(spatial_period, q_orders)
        Iq = self.sins_sinc_intensity_pattern1D(xq, spatial_period, Nsteps, phase_wrap)
        Itot = Iq.sum()
        eta_q = Iq/Itot
        
        c2_rms = self.get_c2_from_sawtooth(spatial_period, phase_wrap)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(x, I, '-k', label = 'sinc2sinc2/sinc2')
        plt.plot(xq, Iq, 'ro', label = 'comb')
        plt.legend(loc = 'best')
        plt.grid('--', alpha = 0.3)
        plt.title('Diffraction pattern:'+'\t'+'$\Phi_{max}= %g \pi$'%M +'\t' +'$\Lambda = %g m$'%spatial_period + '\t' + '$c_2 = %g m$'%c2_rms)
        plt.xlabel(r'$x$'+' '+'[m]')
        plt.ylabel(r'$Intensity$')
        
        plt.figure()
        plt.clf()
        plt.bar(q_orders, eta_q, align = 'center', color ='r')
        plt.title('Diffraction efficiency:'+'\t'+'$\Phi_{max}= %g \pi$'%M +'\t' +'$\Lambda = %g m$'%spatial_period+ '\t' + '$c_2 = %g m$'%c2_rms)
        plt.grid('--', alpha = 0.3)
        plt.xlabel(r'$q$'+' '+'order')
        plt.ylabel(r'$\eta_{q} = I_q/\Sigma I_{q}$')
        plt.xticks(q_orders)