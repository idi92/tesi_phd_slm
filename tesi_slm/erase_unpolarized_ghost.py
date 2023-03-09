import numpy as np 
from tesi_slm.sharp_psf_on_camera import create_devices, SharpPsfOnCamera


class GhostEraser():
    
    def __init__(self):
        cam, mirror = create_devices()
        self._spoc = SharpPsfOnCamera(cam, mirror)
    
    
    def measure_ghost_ratio(self, texp, bg_threshold = 40):
        
        self._spoc._cam.setExposureTime(texp)
        self._spoc.set_slm_flat()
        
        ima_flat = self._spoc._cam.getFutureFrames(1, 50).toNumpyArray()
        bg = ima_flat[ima_flat < bg_threshold].mean()
        clean_flat = ima_flat - bg
        
        self._spoc._write_zernike_on_slm([7000e-9])
        tilt_ima = self._spoc._cam.getFutureFrames(1, 50).toNumpyArray()
        bg = tilt_ima[tilt_ima < bg_threshold].mean()
        clean_tilt = tilt_ima - bg
        self._spoc.set_slm_flat()
        
        peak, yg, xg = self._get_image_peak_and_coords(clean_flat)
        self._ghost_roi = self._cut_image_around_coord(clean_tilt, yg, xg)
        ghost_peak = self._ghost_roi.max()
        tilt_peak, yt, xt = self._get_image_peak_and_coords(clean_tilt)
        self._tilt_roi = self._cut_image_around_coord(clean_tilt, yt, xt)
        print('Tilt ROI (max/sum_roi):')
        print(tilt_peak/self._tilt_roi.sum())
        print('Ghost ROI (max/sum_roi):')
        print(ghost_peak/self._ghost_roi.sum())
        
        ghost_ratio = self._ghost_roi.sum()/self._tilt_roi.sum()
        print('ghost/tilt')
        print(ghost_ratio)
        
        self._clean_flat = clean_flat
        self._clean_tilt = clean_tilt
        return ghost_ratio
        
    def _get_image_peak_and_coords(self, image):
        peak = image.max()
        ymax, xmax = np.where(image==peak)[0][0], np.where(image==peak)[1][0]
        return peak, ymax, xmax
    
    def _cut_image_around_coord(self, image, yc, xc):
        cut_image = image[yc-20:yc+21, xc-20:xc+21]
        return cut_image
    
    def close_slm(self):
        self._spoc.close_slm()
    
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
        