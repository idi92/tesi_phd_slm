import numpy as np
from tesi_slm.utils import fits_io

from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet
from arte.types.slopes import Slopes

def demo():
    
    fname_cam = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\data\\red_data\\240730tilted_psf_on_camera_adjX.fits"
    fname_sh =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\data\\red_data\\240730shwfs_tilt_calib_adjX.fits"
    
    cam_data, cam_header, cam_vect = fits_io.load(fname_cam)
    sh_data, sh_header, sh_vect = fits_io.load(fname_sh)
    
    psf_ref = cam_data[:,:,0]
    
    xbar = cam_header["X_BAR"]
    ybar = cam_header["Y_BAR"]
    
    xref = xbar[0]
    yref = ybar[0]
    
    wf_ref = sh_data[:,:,0]
    