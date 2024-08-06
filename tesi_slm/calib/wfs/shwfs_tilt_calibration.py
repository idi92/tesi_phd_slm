import numpy as np
from tesi_slm.utils import fits_io
import matplotlib.pyplot as plt


from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet
from arte.types.slopes import Slopes


def get_wf_reference():
    
    #fname_cam = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\data\\red_data\\240730tilted_psf_on_camera_adjX.fits"
    fname_sh =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\data\\red_data\\240730shwfs_tilt_calib_adjX.fits"
    
    #cam_data, cam_header, cam_vect = fits_io.load(fname_cam)
    sh_data, sh_header, sh_vect = fits_io.load(fname_sh)
    
    #psf_ref = cam_data[:,:,0]
    
    #xbar = cam_header["X_BAR"]
    #ybar = cam_header["Y_BAR"]
    
    #xref = xbar[0]
    #yref = ybar[0]
    
    wf_ref = sh_data[:,:,0]
    
    return wf_ref
    
    
class SubapertureGridInitialiser():
    
    def __init__(self, wf_reference, pixel_per_sub, Nsub, centroid_threshold=0):
        
        self._wf_ref = wf_reference
        self._pixel_per_sub = int(pixel_per_sub)
        self._Nsub = int(Nsub)
        self._last_grid_shiftYX = None
        self._centroid_threshold = centroid_threshold
        
    def define_subaperture_set(self, ybll = 400, xbll = 350):
        
        frame_shape = self._wf_ref.shape
        
        bll = np.mgrid[ybll:ybll+self._pixel_per_sub*self._Nsub:self._pixel_per_sub,
            xbll:xbll+self._pixel_per_sub*self._Nsub:self._pixel_per_sub].reshape((2, self._Nsub*self._Nsub)).T
    
        self._subaps = ShSubapertureSet.createMinimalSet(
            np.arange(self._Nsub*self._Nsub), frame_shape, self._pixel_per_sub, bll)
    
        self._subaps.update_fix_threshold(self._centroid_threshold)

        self._sc = PCSlopeComputer(self._subaps)
        self._sc.set_frame(self._wf_ref)
    
    def shift_subaperture_grid_with_memory(self, grid_shiftYX=[0, 0]):
        '''
        maybe useless
        '''

        if self._last_grid_shiftYX is not None:
            reset_shiftYX = [-self._last_grid_shiftYX[0], -
                             self._last_grid_shiftYX[1]]
            self._subaps.shiftSubap(self._subaps.keys(), reset_shiftYX)

        self._last_grid_shiftYX = grid_shiftYX

        self._subaps.shiftSubap(self._subaps.keys(), grid_shiftYX)
        # self._last_grid_shiftYX = grid_shiftYX
        self.show_subaperture_grid()
        plt.title(
            f"Grid Shift: Y, X = [{grid_shiftYX[0]} , {grid_shiftYX[1]}]")

    def shift_subaperture_grid(self, grid_shiftYX=[0, 0]):

        self._subaps.shiftSubap(self._subaps.keys(), grid_shiftYX)
        self.show_subaperture_grid()
        plt.title(f"Grid Shift: Y, X = [{grid_shiftYX[0]} , {grid_shiftYX[1]}]")

    def shift_subaperture_grid_to_null_tilt(self):
        offset_x = 42
        offset_y = 42

        while offset_x != 0 or offset_y != 0:
            offset_y = round(self._sc.slopes()[
                             :, 1].mean()/2*self._pixel_per_sub)
            offset_x = round(self._sc.slopes()[
                             :, 0].mean()/2*self._pixel_per_sub)
            self.shift_subaperture_grid([offset_y, offset_x])
            self._sc._reset_all_computed_attributes()

    def update_subapertures_threshold(self, threshold):
        
        for i in self._subaps.values():
            i.setFixThreshold(threshold)
    
    def remove_low_flux_subaperturers(self, threshold = None):
        
        self._sc.remove_low_flux_subaps(threshold)
    
    def show_subaperture_grid(self):
        
        self._show_map(self._sc.subapertures_map()*1000+self._sc.frame())
    
    def show_reference_wf(self):
        
        plt.figure()
        plt.clf()
        plt.imshow(self._wf_ref)
        plt.colorbar(label='ADU')
        plt.title("Reference")
        
    def show_subaperture_flux_histogram(self):
        
        sub_flux = self._sc.subapertures_flux_map().flatten()
        bins = range(0, int(sub_flux.max()),int(sub_flux.max()*0.02))
        plt.figure()
        plt.clf()
        plt.hist(sub_flux, bins, fc='k', ec='k')
        plt.xlabel('Sum Flux [ADU]')
        plt.ylabel('N')
    
    def show_subaperture_flux_map(self):
        
        self._show_map(self._sc.subapertures_flux_map())
    
    def show_slopes_maps(self):
        
        vmin = np.array([self._sc.slopes_x_map().min(), self._sc.slopes_y_map().min()]).min()
        vmax = np.array([self._sc.slopes_x_map().max(), self._sc.slopes_y_map().max()]).max()
               
        plt.subplots(1, 2, sharex=True, sharey=True)
        
        plt.subplot(1, 2, 1)
        plt.title("Slope X")
        plt.imshow(self._sc.slopes_x_map(), vmin = vmin, vmax = vmax)
        
        
        plt.subplot(1, 2, 2)
        plt.title("Slope Y")
        plt.imshow(self._sc.slopes_y_map(), vmin = vmin, vmax = vmax)
        plt.colorbar(label='Slopes units')
        
    @property
    def get_wf_reference(self):
        
        return self._wf_ref
    
    @property
    def get_number_of_subapertures(self):
        
        return self._Nsub
    
    @property
    def get_pixel_per_subapertures(self):
        
        return self._pixel_per_sub
    
    @property
    def get_subapertures(self):
        
        return self._subaps
    
    @property
    def get_slope_computer(self):
        
        return self._sc
    
    def _show_map(self, frame):
        
        plt.figure()
        plt.clf()
        plt.imshow(frame)
        plt.colorbar()


def main(wf_ref=None, corner_xy=(0, 0), nsubaps=50, flux_threshold=57000):
    if wf_ref is None:
        wf_ref = get_wf_reference()
    pixel_per_sub = 26
    # nsubaps = 78  # 50
    sgi = SubapertureGridInitialiser(
        wf_ref, pixel_per_sub, nsubaps, centroid_threshold=70)
    
    sgi.show_reference_wf()
    #top left coords to be fair
    ybll = corner_xy[1]  # 400
    xbll = corner_xy[0]  # 350
    sgi.define_subaperture_set(ybll, xbll)
    
    sgi.show_subaperture_grid()
    
    # sgi.shift_subaperture_grid_to_null_tilt()
    
    sgi.show_subaperture_flux_histogram()

    sgi.remove_low_flux_subaperturers(flux_threshold)
    sgi.show_subaperture_flux_map()
    sgi.show_subaperture_grid()
    sgi.show_slopes_maps()
    return sgi
