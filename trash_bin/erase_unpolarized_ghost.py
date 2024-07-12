import numpy as np 
#from tesi_slm.sharp_psf_on_camera import create_devices, SharpPsfOnCamera
from astropy.io import fits 

class GhostEraserTRASHME():
    
    def __init__(self, spoc):
        #cam, mirror = create_devices()
        self._spoc = spoc
    
    
    def measure_ghost_ratio(self, texp, bg_threshold = 50):
        
        self._spoc._cam.setExposureTime(texp)
        self._spoc.set_slm_flat()
        
        ima_flat = self._spoc._cam.getFutureFrames(1, 30).toNumpyArray()
        bg = ima_flat[ima_flat < bg_threshold].mean()
        clean_flat = ima_flat - bg
        #clean_flat[clean_flat<0] = 0
        
        self._spoc.write_zernike_on_slm([7000e-9])
        tilt_ima = self._spoc._cam.getFutureFrames(1, 30).toNumpyArray()
        bg = tilt_ima[tilt_ima < bg_threshold].mean()
        clean_tilt = tilt_ima - bg
        #clean_tilt[clean_tilt<0] = 0
        self._spoc.set_slm_flat()
        
        flat_peak, yg, xg = self._get_image_peak_and_coords(clean_flat)
        self._flat_roi = self._cut_image_around_coord(clean_flat, yg, xg)
        self._ghost_roi = self._cut_image_around_coord(clean_tilt, yg, xg)
        ghost_peak = self._ghost_roi.max()
        tilt_peak, yt, xt = self._get_image_peak_and_coords(clean_tilt)
        self._tilt_roi = self._cut_image_around_coord(clean_tilt, yt, xt)
        # print('Tilt ROI (max/sum_roi):')
        # print(tilt_peak/self._tilt_roi.sum())
        # print('Ghost ROI (max/sum_roi):')
        # print(ghost_peak/self._ghost_roi.sum())
        #

        ghost_ratio = self._ghost_roi.sum()/self._flat_roi.sum()
        modulated_ratio = self._tilt_roi.sum()/self._flat_roi.sum()
        #print('ghost/tilt')
        #print(ghost_ratio)
        
        self._clean_flat = clean_flat
        self._clean_tilt = clean_tilt
        return ghost_ratio, modulated_ratio
        
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_coord(self, image, yc, xc):
        cut_image = image[yc-25:yc+25, xc-25:xc+25]
        return cut_image
    
    def iterate(self, angle, texp=0.05, N_iter=100, bg_threshold=50):
        self._texp  = texp
        self._angle = angle
        ghost_ratio_values = np.zeros(N_iter)
        mod_ratio_values = np.zeros(N_iter)
        
        
        for t in range(N_iter):
            print('iter %d'%t)
            ghost_ratio_values[t], mod_ratio_values[t] = self.measure_ghost_ratio(texp, bg_threshold)
        self._ghost_ratio_mean = ghost_ratio_values.mean()
        self._ghost_ratio_err = ghost_ratio_values.std()
        self._mod_ratio_mean = mod_ratio_values.mean()
        self._mod_ratio_err = mod_ratio_values.std()
        
    
    
    def save(self, fname):
        hdr = fits.Header()
       
        hdr['T_EX_MS'] = self._texp
        hdr['ANG'] = self._angle
        data = np.array([self._ghost_ratio_mean,self._ghost_ratio_err, self._mod_ratio_mean, self._mod_ratio_err])
        fits.writeto(fname, data, hdr)
        # fits.append(fname, self._ghost_ratio_err)
        # fits.append(fname, self._mod_ratio_mean)
        # fits.append(fname, self._mod_ratio_err)
    
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
    
    
    def show_plots(self):
        import matplotlib.pyplot as plt
        plt.subplots(1, 2, sharex=True, sharey=True)
        plt.subplot(1, 2, 1)
        plt.title('tilt')
        plt.imshow(self._clean_tilt, cmap = 'jet')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('flat')
        plt.imshow(self._clean_flat, cmap = 'jet')
        plt.colorbar()
        
        plt.figure()
        plt.title('ghost')
        plt.imshow(self._ghost_roi, cmap = 'jet')
        plt.colorbar()
        plt.figure()
        plt.title('roi')
        plt.imshow(self._tilt_roi, cmap = 'jet')
        plt.colorbar()

class AnalyzeGhostRatioTRASHME():
    
    def __init__(self, angles, fdir):
        Nangles = len(angles)
        self._rot_angle = np.zeros(Nangles)
        self._ghost_mean  = np.zeros(Nangles)
        self._ghost_err  = np.zeros(Nangles)
        self._mod_mean  = np.zeros(Nangles)
        self._mod_err  = np.zeros(Nangles)
        
        for n in range(Nangles):
            fname = fdir+'/230310ge_ang'+'%d.fits'%angles[n]
            texp, angle, data = GhostEraser.load(fname)
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
        