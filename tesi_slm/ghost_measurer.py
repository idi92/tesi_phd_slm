import numpy as np 
from tesi_slm.camera_masters import CameraMastersAnalyzer
from astropy.io import fits

class MeasureGhost():
    
    def __init__(self, spoc, fname_masters):
        self._spoc = spoc
        self._spoc.change_circular_mask(RadiusInPixel=1920*0.5)
        self._texp, self._fNframes, \
         self._master_dark, self._master_background = \
         CameraMastersAnalyzer.load_camera_masters(fname_masters)
    
    def iterate_measure_for_angle(self, angle, N_iter=100):
        
        self._angle = angle
        ghost_ratio_values = np.zeros(N_iter)
        mod_ratio_values = np.zeros(N_iter)
        Nframes = 30
        self._spoc._cam.setExposureTime(self._texp)
        
        for t in range(N_iter):
            print('iter %d'%t)
            self._spoc.set_slm_flat()
            flat_ima = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
            clean_flat = self._get_clean_mean_image(flat_ima)
            
            self._spoc._write_zernike_on_slm([7000e-9])
            tilt_ima = self._spoc._cam.getFutureFrames(Nframes).toNumpyArray()
            clean_tilt = self._get_clean_mean_image(tilt_ima)
            
            flat_roi = self._cut_image_around_coord(clean_flat, 574, 461)
            mod_roi = self._cut_image_around_coord(clean_tilt, 573, 375)
            ghost_roi = self._cut_image_around_coord(clean_tilt, 574, 461)
            ghost_ratio_values[t] = ghost_roi.sum()/flat_roi.sum()
            mod_ratio_values[t] = mod_roi.sum()/flat_roi.sum()
            
        self._ghost_ratio_mean = ghost_ratio_values.mean()
        self._ghost_ratio_err = ghost_ratio_values.std()
        self._mod_ratio_mean = mod_ratio_values.mean()
        self._mod_ratio_err = mod_ratio_values.std()
        
    def _get_clean_mean_image(self, cube_ima):
        Nframes  = cube_ima.shape[-1]
        for n in range(Nframes):
                cube_ima[:, :, n] = cube_ima[:,:,n] - self._master_background \
                 - self._master_dark
        return cube_ima.mean(axis=-1)
    
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_coord(self, image, yc, xc):
        cut_image = image[yc-25:yc+25, xc-25:xc+25]
        return cut_image
    
    def save(self, fname):
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['ANG'] = self._angle
        data = np.array([self._ghost_ratio_mean,self._ghost_ratio_err, self._mod_ratio_mean, self._mod_ratio_err])
        fits.writeto(fname, data, hdr)
        
    @staticmethod    
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        data = hduList[0].data 
        # ghost_ratio_err = hduList[1].data 
        # mod_ratio_mean = hduList[2].data 
        # mod_ratio_err = hduList[3].data 
        
        angle = header['ANG']
        texp = header['T_EX_MS']
        return texp, angle, data
    
class AnalyzeGhostRatio():
    
    def __init__(self, angles, fdir):
        Nangles = len(angles)
        self._rot_angle = np.zeros(Nangles)
        self._ghost_mean  = np.zeros(Nangles)
        self._ghost_err  = np.zeros(Nangles)
        self._mod_mean  = np.zeros(Nangles)
        self._mod_err  = np.zeros(Nangles)
        
        for n in range(Nangles):
            fname = fdir+'/230310ge_ang'+'%d.fits'%angles[n]
            texp, angle, data = MeasureGhost.load(fname)
            self._rot_angle[n] = angle
            self._ghost_mean[n] = data[0]
            self._ghost_err[n] = data[1]
            self._mod_mean[n] = data[2]
            self._mod_err[n] = data[3]
    
    def show_ratio(self):        
        import matplotlib.pyplot as plt
        plt.subplots(2,1,sharex=True)
        plt.subplot(2,1,1)
        plt.plot(self._rot_angle, self._ghost_mean, 'bo-', label = 'ghost ratio')
        plt.errorbar(self._rot_angle, self._ghost_mean, self._ghost_err, fmt='.b')
        plt.xlabel('angle [deg]')
        plt.ylabel('$<\Sigma ghost(x_i,y_j)_{roi}> / <\Sigma flat(x_i, y_j)_{roi}>$')
        plt.legend(loc='best')
        plt.grid(ls ='--', alpha = 0.5)
        plt.subplot(2,1,2)
        plt.plot(self._rot_angle, self._mod_mean, 'ro-', label = 'mod ratio')
        plt.errorbar(self._rot_angle, self._mod_mean, self._mod_err, fmt='.r')
        plt.xlabel('angle [deg]')
        plt.ylabel('$<\Sigma modulated(x_i,y_j)_{roi}> / <\Sigma flat(x_i, y_j)_{roi}>$')
        plt.legend(loc='best')
        plt.grid(ls ='--', alpha = 0.5)