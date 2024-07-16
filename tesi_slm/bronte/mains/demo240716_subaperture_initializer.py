import numpy as np
import matplotlib.pyplot as plt

from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet
from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.rebin import rebin


from tesi_slm.utils import fits_io
import datetime as date




def demo_subaperture_setup():
    
    frame = _load_file()
    
    _show_map(frame)
    
    pixel_size = 5.5e-6 
    lens_size = 144e-6
    pixel_per_sub = int(lens_size/pixel_size)
    frame_shape = frame.shape
    
    Nsub = 50
    
    #selecting bottom left coordinates for bl list
    #actually is top left, depends how you plot the map
    
    ybll = 400
    xbll = 360
    subaps, sc = _define_subaperute_set(frame, ybll, xbll, Nsub, pixel_per_sub)
    
    #shift subapgrid to center the spots
    
    subshiftYX = [8,-6]
    _shift_subaperture_grid(subaps, sc, subshiftYX)
    
    #setting a threshold for the subaps looking at the flux
    _show_map(sc.subapertures_flux_map())
    sub_flux = sc.subapertures_flux_map().flatten()
    plt.figure()
    plt.clf()
    plt.hist(sub_flux, bins=range(0, int(sub_flux.max()), int(sub_flux.max()*0.02)), fc='k', ec='k')
    plt.xlabel('Sum Flux [ADU]')
    plt.ylabel('N')
    
    sc.remove_low_flux_subaps(threshold=45000)
    _show_map(sc.subapertures_map()*1000+sc.frame())
    _show_map(sc.subapertures_flux_map())

def _load_file():
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240716_bronte_subaperture_setting\\"
    fname = fpath + "240716shwfs_pupil_red.fits"
    
    frame, header_dict, vector_dict = fits_io.load(fname) 
    
    return frame

def _show_map(frame):
    
    plt.figure()
    plt.clf()
    plt.imshow(frame)
    plt.colorbar()

def _define_subaperute_set(frame, ybll = 400, xbll = 360, Nsub = 50, pixel_per_sub = 26):
    
    frame_shape = frame.shape
    bll = np.mgrid[ybll:ybll+pixel_per_sub*Nsub:pixel_per_sub,
                    xbll:xbll+pixel_per_sub*Nsub:pixel_per_sub].reshape((2, Nsub*Nsub)).T
    
    subaps = ShSubapertureSet.createMinimalSet(
        np.arange(Nsub*Nsub), frame_shape, pixel_per_sub, bll)
    
    sc = PCSlopeComputer(subaps)
    sc.set_frame(frame)
    _show_map(sc.subapertures_map()*1000+sc.frame())
    
    return subaps, sc

def _shift_subaperture_grid(subaps, sc, subshiftYX=[0,0]):
    
    subaps.shiftSubap(subaps.keys(), subshiftYX)
    _show_map(sc.subapertures_map()*1000 + sc.frame())